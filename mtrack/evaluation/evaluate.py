import numpy as np

from comatch import match_components
from mtrack.preprocessing import nml_to_g1
from mtrack.evaluation.voxel_skeleton import VoxelSkeleton
from mtrack.evaluation.matching_graph import MatchingGraph


def build_matching_graph(tracing_g1, reconstruction_g1, voxel_size, distance_threshold, subsample):
    """
    tracing: g1 graph in voxel space.

    reconstruction: g1 graph in voxel space.

    voxel_size: size of one voxel, 3d.

    distance_threshold: maximal distance above which two vertices can be matched to each other

    subsample: Reduces the number of points to match in a microtubule by that factor. I.e every subsample voxel 
               is retained. If the number exceeds the number of voxels in a track only the start and end 
               voxel are retained.
    """

    # Get individual microtubule paths:
    tracing_mts = tracing_g1.get_components(min_vertices=1, 
                                            output_folder=None,
                                            return_graphs=True)

    reconstruction_mts = reconstruction_g1.get_components(min_vertices=1, 
                                                          output_folder=None,
                                                          return_graphs=True)

    # Interpolate linearly between points and subsample
    # by conversion to vertex skeletons:
    tracing_vertex_skeletons = []
    reconstruction_vertex_skeletons = []
    for tracing_mt in tracing_mts:
        vs = VoxelSkeleton(tracing_mt, voxel_size=voxel_size, verbose=False, subsample=subsample)
        tracing_vertex_skeletons.append(vs)

    for reconstruction_mt in reconstruction_mts:
        vs = VoxelSkeleton(reconstruction_mt, voxel_size=voxel_size, verbose=False, subsample=subsample)
        reconstruction_vertex_skeletons.append(vs)

    # Construct matching graph:
    matching_graph = MatchingGraph(tracing_vertex_skeletons,
                                   reconstruction_vertex_skeletons,
                                   distance_threshold,
                                   voxel_size,
                                   verbose=False,
                                   distance_cost=True,
                                   initialize_all=True)

    return matching_graph


def evaluate_matching_graph(matching_graph, use_distance_costs=True, max_edges=1, export_to=None):
    nodes_gt, nodes_rec, edges_gt_rec, labels_gt, labels_rec, edge_costs, edge_conflicts, edge_pairs = matching_graph.export_to_comatch()
    if not use_distance_costs:
        edge_costs = None
    if not edge_conflicts:
        edge_conflicts = None

    try:
        # Quadmatch
        print "Using quadmatch..."
        label_matches, node_matches, num_splits, num_merges, num_fps, num_fns = match_components(nodes_gt, nodes_rec, 
                                                                                                 edges_gt_rec, labels_gt, labels_rec, 
                                                                                                 edge_conflicts=edge_conflicts,
                                                                                                 max_edges=max_edges, edge_costs=edge_costs)
    except TypeError:
        # Comatch
        if max_edges > 1:
            allow_many_to_many = True
        else:
            allow_many_to_many = False

        if use_distance_costs:
            no_match_cost = max(edge_costs) * 2
        else:
            no_match_cost = 0.0

        print "Using comatch with no match cost: {}".format(no_match_cost)

        label_matches, node_matches, num_splits, num_merges, num_fps, num_fns = match_components(nodes_gt, nodes_rec, 
                                                                                                 edges_gt_rec, labels_gt, labels_rec, 
                                                                                                 allow_many_to_many=(max_edges>1), edge_costs=edge_costs,
                                                                                                 no_match_costs=no_match_cost)

    matching_graph.import_matches(node_matches)
    topological_errors = {"splits": num_splits, "merges": num_merges, "fps": num_fps, "fns": num_fns}
    node_errors = matching_graph.evaluate()

    if export_to is not None:
        matching_graph.export_all(export_to)

    return matching_graph, topological_errors, node_errors
