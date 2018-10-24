import numpy as np
import os

from comatch import match_components
from mtrack.preprocessing import nml_to_g1, g1_to_nml
from mtrack.evaluation.voxel_skeleton import VoxelSkeleton
from mtrack.evaluation.matching_graph import MatchingGraph
from mtrack.graphs import G1
import json


def evaluate(tracing, 
             reconstruction, 
             voxel_size, 
             distance_threshold, 
             subsample, 
             use_distance_costs=True, 
             max_edges=1,
             export_to=None,
             x_lim=None,
             y_lim=None,
             z_lim=None,
             optimality_gap=0.0,
             absolute=True,
             time_limit=None):
    """
    tracing and reconstruction can be provided in either .nml file
    or g1 graph. In both cases voxel scaling is required and any knossos offset has
    to be corrected prior to this.

    Limits need to provided in voxel units.

    distance_threshold should be given in physical coordinates.
    """

    inputs = [tracing, reconstruction]
    inputs_g1 = []

    # Convert inputs to g1 graphs
    for item in inputs:
        if isinstance(item, str):
            if (not item.endswith(".nml")) or (not os.path.isfile(item)):
                print item, item.endswith(".nml"), os.path.isfile(item)
                raise ValueError("Provide inputs as path to knossos nml file or as a g1 graph directly")
            else:
                inputs_g1.append(nml_to_g1(item, output_file=None))
        elif isinstance(tracing, G1):
            inputs_g1.append(item)
        else:
            raise ValueError("Input not recognized. Provide nml file or g1 graph")


    tracing_g1 = inputs_g1[0]
    reconstruction_g1 = inputs_g1[1]

    # Cut tracing and rec to roi
    if x_lim is not None:
        if (y_lim is None) or (z_lim is None):
            raise ValueError("Provide all limits or none.")
        else:
            tracing_g1 = cut_to_roi(tracing_g1, x_lim, y_lim, z_lim, 
                                    export_to=export_to + "/tracing_cut.nml")

            reconstruction_g1 = cut_to_roi(reconstruction_g1, x_lim, y_lim, z_lim, 
                                           export_to=export_to + "/reconstruction_cut.nml")

    matching_graph = build_matching_graph(tracing_g1, reconstruction_g1, 
                                          voxel_size, distance_threshold, 
                                          subsample)

    matching_graph, topological_errors, node_errors = evaluate_matching_graph(matching_graph, 
                                                                              use_distance_costs, 
                                                                              max_edges, export_to,
                                                                              optimality_gap, absolute,
                                                                              time_limit)

    return matching_graph, topological_errors, node_errors


def build_matching_graph(tracing_g1, reconstruction_g1, voxel_size, distance_threshold, subsample):
    """
    tracing: g1 graph in voxel space.

    reconstruction: g1 graph in voxel space. 

    voxel_size: size of one voxel, 3d.

    distance_threshold: maximal distance above which two vertices can be matched to each other

    subsample: Reduces the number of points to match in a path by that factor. I.e every subsample voxel 
               is retained. If the number exceeds the number of voxels in a track only the start and end 
               voxel are retained.
    """

    # Get individual microtubule paths:
    tracing_mts = tracing_g1.get_components(min_vertices=2, 
                                            output_folder=None,
                                            return_graphs=True)

    reconstruction_mts = reconstruction_g1.get_components(min_vertices=2, 
                                                          output_folder=None,
                                                          return_graphs=True)

    # Interpolate linearly between points and subsample
    # by conversion to vertex skeletons:
    tracing_vertex_skeletons = []
    reconstruction_vertex_skeletons = []
    print "Construct voxel skeletons..."
    for tracing_mt in tracing_mts:
        vs = VoxelSkeleton(tracing_mt, voxel_size=voxel_size, verbose=False, subsample=subsample)
        tracing_vertex_skeletons.append(vs)

    for reconstruction_mt in reconstruction_mts:
        vs = VoxelSkeleton(reconstruction_mt, voxel_size=voxel_size, verbose=False, subsample=subsample)
        reconstruction_vertex_skeletons.append(vs)

    # Construct matching graph:
    print "Construct matching graph..."
    matching_graph = MatchingGraph(tracing_vertex_skeletons,
                                   reconstruction_vertex_skeletons,
                                   distance_threshold,
                                   voxel_size,
                                   verbose=True,
                                   distance_cost=True,
                                   initialize_all=True)

    return matching_graph


def evaluate_matching_graph(matching_graph, use_distance_costs=True, max_edges=1, export_to=None, optimality_gap=0.0, absolute=True, time_limit=None):
    if max_edges>1:
        edge_conflicts = True
    else:
        edge_conflicts = False

    nodes_gt, nodes_rec, edges_gt_rec, labels_gt, labels_rec, edge_costs, edge_conflicts, edge_pairs = matching_graph.export_to_comatch(edge_conflicts=edge_conflicts, 
                                                                                                                                        edge_pairs=False)

    if not use_distance_costs:
        edge_costs = None

    try:
        # Quadmatch
        print "Match using quadmatch..."
        label_matches, node_matches, num_splits, num_merges, num_fps, num_fns = match_components(nodes_gt, nodes_rec, 
                                                                                                 edges_gt_rec, labels_gt, labels_rec, 
                                                                                                 edge_conflicts=edge_conflicts,
                                                                                                 max_edges=max_edges, edge_costs=edge_costs,
                                                                                                 optimality_gap=optimality_gap, absolute=absolute,
                                                                                                 time_limit=time_limit)
    except TypeError:
        # Comatch
        if use_distance_costs:
            no_match_cost = max(edge_costs) * 100
        else:
            no_match_cost = 0

        print "Failed... Fallback to comatch with no match cost: {}".format(no_match_cost)

        label_matches, node_matches, num_splits, num_merges, num_fps, num_fns = match_components(nodes_gt, nodes_rec, 
                                                                                                 edges_gt_rec, labels_gt, labels_rec, 
                                                                                                 allow_many_to_many=(max_edges>1), edge_costs=edge_costs,
                                                                                                 no_match_costs=no_match_cost, 
                                                                                                 optimality_gap=optimality_gap,
                                                                                                 absolute=absolute,
                                                                                                 time_limit=time_limit)

    matching_graph.import_matches(node_matches)
    topological_errors = {"splits": num_splits, "merges": num_merges, "fps": num_fps, "fns": num_fns}
    node_errors = matching_graph.evaluate()

    if export_to is not None:
        matching_graph.export_all(export_to)
        with open(export_to + "/object_stats.txt", "w+") as f:
            json.dump(topoligical_errors, f)


    return matching_graph, topological_errors, node_errors


def cut_to_roi(g1, x_lim, y_lim, z_lim, export_to=None):
    """
    Limits need to be provided in voxel coordinates.
    The database limits are given in physical coordinates so this has
    to be converted before.
    """
    min_roi = np.array([x_lim["min"], y_lim["min"], z_lim["min"]])
    max_roi = np.array([x_lim["max"], y_lim["max"], z_lim["max"]])
	
    in_roi_vp = g1.new_vertex_property("in_roi", "bool", False)
    for v in g1.get_vertex_iterator():
	position = g1.get_position(v)
	if np.all(position>=min_roi) and np.all(position<=max_roi):
	    in_roi_vp[v] = True
	else:
	    in_roi_vp[v] = False
    
    g1.set_vertex_filter(in_roi_vp)
    g1.purge_vertices()
    g1.purge_edges()
    if export_to is not None:
        g1_to_nml(g1, export_to, knossify=True)

    return g1
