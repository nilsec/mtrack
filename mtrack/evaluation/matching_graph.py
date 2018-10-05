import numpy as np
from scipy.spatial import KDTree

from mtrack.graphs import G1
from mtrack.preprocessing import g1_to_nml

class MatchingGraph(object):
    def __init__(self, 
                 groundtruth_skeletons, 
                 reconstructed_skeletons, 
                 distance_threshold,
                 voxel_size,
                 verbose=False, 
                 initialize_all=True):
        """
        Matching graph representation.
        """
        self.skeletons = {"gt": groundtruth_skeletons, 
                          "rec": reconstructed_skeletons}

        self.graphs = {"gt": [skeleton.get_graph() for skeleton in groundtruth_skeletons], 
                       "rec": [skeleton.get_graph() for skeleton in reconstructed_skeletons]}

        self.verbose = verbose

        self.distance_threshold = distance_threshold

        self.voxel_size = voxel_size

        self.total_vertices = self.__get_total_vertices()
        
        """
        Generate various mappings and indexing schemes: 

        gt_mappings = {"ids": ids_gt, "positions": positions_gt,
                       "id_to_mvertex": ids_gt_to_mvertex, "mvertex_to_id": mvertex_to_ids_gt}

        rec_mappings = {"ids": ids_rec, "positions": positions_rec,
                        "id_to_mvertex": ids_rec_to_mvertex, "mvertex_to_id": mvertex_to_ids_rec}

        """
        if initialize_all:
            self.matching_graph, self.mappings, self.mv_to_v, self.v_to_mv = self.__initialize()
            self.__add_skeleton_edges()
            self.__add_matching_edges(distance_threshold, voxel_size)


        # All further initializations should happen here and not before the call to __initialize():


    def __initialize(self):
        mv_to_v = {}
        v_to_mv = {}

        mv_gt = []
        positions_gt = []

        mv_rec = []
        positions_rec = []

        mappings = {"gt": {"mv_ids": mv_gt, "positions": positions_gt},
                    "rec": {"mv_ids": mv_rec, "positions": positions_rec}}

        matching_graph = G1(self.total_vertices)

        matching_graph.new_edge_property("is_skeleton", dtype="bool", value=False)
        matching_graph.new_edge_property("is_matching", dtype="bool", value=False)

        matching_graph.new_vertex_property("is_gt", dtype="bool", value=False)


        v_matching = 0
        graph_hash = 0
        for tag in ["gt", "rec"]:
            for graph in self.graphs[tag]:
                graph.set_hash(graph_hash)
                for v in graph.get_vertex_iterator():
                    # Create indices to positions for kdtree later
                    position = np.array(graph.get_position(v))
                    mappings[tag]["positions"].append(position)

                    # Initialize vertex in matching graph
                    matching_graph.set_position(v_matching, position)
                    if tag == "gt":
                        matching_graph.set_vertex_property("is_gt", v_matching, True)
                    else:
                        matching_graph.set_vertex_property("is_gt", v_matching, False)

                    mappings[tag]["mv_ids"].append(v_matching)

                    # Build mappings
                    mv_to_v[v_matching] = (graph, int(v))
                    v_to_mv[(graph,int(v))] = v_matching
                    v_matching += 1

                graph_hash += 1

        return matching_graph, mappings, mv_to_v, v_to_mv

    def __get_total_vertices(self):
        total_vertices = 0

        for tag in ["gt", "rec"]:
            for graph in self.graphs[tag]:
                total_vertices += graph.get_number_of_vertices()

        return total_vertices

    def get_number_of_vertices(self):
        return self.matching_graph.get_number_of_vertices()

    def get_number_of_edges(self):
        return self.matching_graph.get_number_of_edges()

    def is_groundtruth_mv(self, mv):
        """
        v: vertex id of matching graph

        Returns the identity of a vertex
        in the matching graph wether it
        is part of the ground truth or 
        not.
        """
        return self.matching_graph.get_vertex_property("is_gt", mv)

    def get_tag_mv(self, mv):
        if self.is_groundtruth_mv(mv):
            tag = "gt"
        else:
            tag = "rec"
        return tag

    def get_skeleton_id(self, mv):
        return self.mv_to_v[mv]

    def get_matching_id(self, graph, v):
        return self.v_to_mv[(graph, v)]

    def get_matching_edge(self, graph, v0, v1):
        return (self.get_matching_id(graph, v0), self.get_matching_id(graph, v1))
   
    def get_positions(self, tag):
        return self.mappings[tag]["positions"]

    def get_mv_ids(self, tag):
        return self.mappings[tag]["mv_ids"]

    def add_edge(self, u, v):
        return self.matching_graph.add_edge(u,v)

    def set_matching(self, e):
        self.matching_graph.set_edge_property("is_matching", None, None, True, e=e)

    def set_skeleton(self, e):
        self.matching_graph.set_edge_property("is_skeleton", None, None, True, e=e)

    def get_edge_type(self, e):
        if self.matching_graph.get_edge_property("is_matching", e.source(), e.target()):
            return "matching"

        elif self.matching_graph.get_edge_property("is_skeleton", e.source(), e.target()):
            return "skeleton"

        else:
            raise ValueError("Edge has no type. Initialization failed.")

    def mask_skeleton_edges(self):
        self.matching_graph.set_edge_filter(None)
        skeleton_ep = self.matching_graph.get_edge_property("is_skeleton")
        self.matching_graph.set_edge_filter(skeleton_ep)

    def mask_matching_edges(self):
        self.matching_graph.set_edge_filter(None)
        matching_ep = self.matching_graph.get_edge_property("is_matching")
        self.matching_graph.set_edge_filter(matching_ep)

    def clear_edge_masks(self):
        self.matching_graph.set_edge_filter(None)

    def export_to_comatch(self):
        nodes_gt = []
        nodes_rec = []
        edges_gt_rec = []

        for v in self.matching_graph.get_vertex_iterator():
            if self.is_groundtruth_mv(v):
                nodes_gt.append(int(v))
            else:
                nodes_rec.append(int(v))

        self.mask_matching_edges()
        for e in self.matching_graph.get_edge_iterator():
            edge_gt_rec = []
            if self.is_groundtruth_mv(e.source()):
                edge_gt_rec = [int(e.source()), int(e.target())]
            else:
                edge_gt_rec = [int(e.target()), int(e.source())]

            edges_gt_rec.append(tuple(edge_gt_rec))

        return nodes_gt, nodes_rec, edges_gt_rec

    def to_nml(self, path):
        g1_to_nml(self.matching_graph, path, voxel_size=self.voxel_size, knossify=True)

    def __add_skeleton_edges(self):
        """
        Add original skeleton edges back to the matching graph.
        """
        if self.verbose:
            print "Add skeleton edges..."

        for tag in ["gt", "rec"]:
            graphs = self.graphs[tag]
            for i in range(len(graphs)):
                for e in graphs[i].get_edge_iterator():
                    matching_edge = self.get_matching_edge(graphs[i], e.source(), e.target())
                    edge = self.add_edge(*matching_edge)
                    self.set_skeleton(edge)

    def __add_matching_edges(self, distance_threshold, voxel_size):
        """
        Connect ground truth and reconstruction vertices of the matching
        graph to each other that are below a certain distance threshold.
        Distance threshold should be given in physical coordinates 
        together with the voxel size.
        """

        if self.verbose:
            print "Add matching edges..."

        gt_positions = self.get_positions("gt")
        rec_positions = self.get_positions("rec")

        # tag_positions[i] == matching_graph.get_position[tag_mv_ids[i]]
        gt_mv_ids = self.get_mv_ids("gt")
        rec_mv_ids = self.get_mv_ids("rec")

        if self.verbose:
            print "Initialize KDTrees..."
        gt_tree = KDTree(gt_positions * np.array(voxel_size))
        rec_tree = KDTree(rec_positions * np.array(voxel_size))

        """
        From the docs:
        KDTree.query_ball_tree(other, r, p=2.0, eps=0)
        For each element self.data[i] of this tree, 
        results[i] is a list of the indices of its neighbors in other.data.
        """
        if self.verbose:
            print "Query ball tree..."
        results = gt_tree.query_ball_tree(rec_tree, r=distance_threshold)
       
        if self.verbose:
            print "Add matching edges to graph..."
        for gt_id in range(len(results)):
            mv_id_source = gt_mv_ids[gt_id]

            for rec_id in results[gt_id]:
                mv_id_target = rec_mv_ids[rec_id]
                edge = self.add_edge(mv_id_source, mv_id_target)
                # Set matching edge property:
                self.set_matching(edge)
