import numpy as np
import os
from scipy.spatial import KDTree
import json
import logging

from mtrack.graphs import G1
from mtrack.preprocessing import g1_to_nml

logger = logging.getLogger(__name__)

class MatchingGraph(object):
    def __init__(self, 
                 groundtruth_skeletons, 
                 reconstructed_skeletons, 
                 distance_threshold,
                 voxel_size,
                 distance_cost=True,
                 initialize_all=True):
        """
        TODO: Clean up export/mask methods - too much code duplication now.
              Handle conflict find during import, unnecessary slow now

        Matching graph representation.

        groundtruth_skeletons: list of groundtruth voxel skeletons in voxel coordinates.
        reconstructed_skeletons: list of reconstructed voxel_skeletons in voxel_coordinates.
        distance_threshold: should correspond to physical positions if voxel_size != [1,1,1].
        """
        self.skeletons = {"gt": groundtruth_skeletons, 
                          "rec": reconstructed_skeletons}

        self.graphs = {"gt": [skeleton.get_graph() for skeleton in groundtruth_skeletons], 
                       "rec": [skeleton.get_graph() for skeleton in reconstructed_skeletons]}

        self.distance_cost = distance_cost

        self.distance_threshold = distance_threshold

        self.voxel_size = voxel_size

        self.total_vertices = self.__get_total_vertices()

        self.dummy_vertex = -1

        if initialize_all:
            self.matching_graph, self.mappings, self.mv_to_v, self.v_to_mv = self.__initialize()
            self.__add_skeleton_edges()
            self.__add_matching_edges(distance_threshold, voxel_size)


        # All further initializations should happen here and not before the call to __initialize():
        self.matched = False


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

        matching_graph.new_edge_property("distance", dtype="float", value=0.0)

        matching_graph.new_vertex_property("is_gt", dtype="bool", value=False)
        matching_graph.new_vertex_property("label", dtype="int", value=-1)


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

                    matching_graph.set_vertex_property("label", v_matching, graph_hash)
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

    def set_distance(self, e):
        pos_source = np.array(self.matching_graph.get_position(e.source()))
        pos_target = np.array(self.matching_graph.get_position(e.target()))
        distance = np.linalg.norm(pos_source - pos_target)
        self.matching_graph.set_edge_property("distance", e.source(), e.target(), distance)

    def get_distance(self, e):
        distance = self.matching_graph.get_edge_property("distance", e.source(), e.target())
        return distance

    def set_skeleton(self, e):
        self.matching_graph.set_edge_property("is_skeleton", None, None, True, e=e)

    def get_edge_type(self, e):
        if self.matching_graph.get_edge_property("is_matching", e.source(), e.target()):
            return "matching"

        elif self.matching_graph.get_edge_property("is_skeleton", e.source(), e.target()):
            return "skeleton"

        else:
            raise ValueError("Edge has no type. Initialization failed.")

    def __mask_vp(self, vp_name):
        vp = self.matching_graph.get_vertex_property(vp_name)
        self.matching_graph.set_vertex_filter(vp)

    def __mask_ep(self, ep_name):
        ep = self.matching_graph.get_edge_property(ep_name)
        self.matching_graph.set_edge_filter(ep)

    def mask_skeleton_edges(self):
        self.__mask_ep("is_skeleton")

    def mask_matching_edges(self):
        self.__mask_ep("is_matching")

    def mask_groundtruth(self):
        self.__mask_vp("is_gt")

    def mask_reconstruction(self):
        groundtruth_vp = self.matching_graph.get_vertex_property("is_gt")
        reconstruction_vp = self.matching_graph.new_vertex_property("is_rec", dtype="bool",  value=np.logical_not(groundtruth_vp.a))
        self.matching_graph.set_vertex_filter(reconstruction_vp)

    def mask_tps(self):
        self.__mask_vp("tp")

    def mask_fns(self):
        self.__mask_vp("fn")

    def mask_fps(self):
        self.__mask_vp("fp")

    def mask_splits(self):
        self.__mask_ep("split")
        self.__mask_vp("split")

    def mask_mergers(self):
        self.__mask_ep("merge")
        self.__mask_vp("merge")

    def __mark_truefalse_vertices(self):
        if not self.matched:
            raise ValueError("Graph not matched yet. Import comatch results before analyzing.")

        self.clear_vertex_masks()
        self.clear_edge_masks()

        self.matching_graph.new_vertex_property("fp", dtype="bool", value=False)
        self.matching_graph.new_vertex_property("fn", dtype="bool", value=False)
        self.matching_graph.new_vertex_property("tp", dtype="bool", value=False)
        self.mask_matched_edges()

        for v in self.matching_graph.get_vertex_iterator():
            incident_edges = list(self.matching_graph.get_incident_edges(v))
            if not incident_edges:
                if self.is_groundtruth_mv(v):
                    self.matching_graph.set_vertex_property("fn", v, True)
                else:
                    self.matching_graph.set_vertex_property("fp", v, True)
            else:
                self.matching_graph.set_vertex_property("tp", v, True)
                self.matching_graph.set_vertex_property("fn", v, False)
                self.matching_graph.set_vertex_property("fp", v, False)

        self.clear_edge_masks()

    def __is_vp(self, vp_name, mv):
        return self.matching_graph.get_vertex_property(vp_name, mv)

    def is_tp(self, mv):
        return self.__is_vp("tp", mv)

    def is_fp(self, mv):
        return self.__is_vp("fp", mv)

    def is_fn(self, mv):
        return self.__is_vp("fn", mv)

    def __is_ep(self, ep_name, e):
        return self.matching_graph.get_edge_property(ep_name, e.source(), e.target())

    def is_split(self, e):
        return self.__is_ep("split", e)

    def is_merge(self, e):
        return self.__is_ep("merge", e)

    def get_nbs(self, mv, subgraph):
        if subgraph not in ["matched", "skeleton", "matching"]:
            raise ValueError

        if subgraph == "matched":
            self.mask_matched_edges()
        elif subgraph == "skeleton":
            self.mask_skeleton_edges()
        else:
            self.mask_matching_edges()

        incident_edges_mv = self.matching_graph.get_incident_edges(mv)
      
        nbs_mv = set()
        if incident_edges_mv:
            for e in incident_edges_mv:
                nbs_mv.add(e.source())
                nbs_mv.add(e.target())

            nbs_mv.remove(mv)

        self.clear_edge_masks()

        return list(nbs_mv)

    def get_label(self, mv):
        if mv == self.dummy_vertex:
            return -1
        else:
            return int(self.matching_graph.get_vertex_property("label", mv))

    def get_vertex_iterator(self):
        return self.matching_graph.get_vertex_iterator()

    def get_edge_iterator(self):
        return self.matching_graph.get_edge_iterator()

    def __get_matched_label(self, matching_nbs):
        component_ids = set()
        if matching_nbs:
            for mv in matching_nbs:
                component_id = self.get_label(mv)
                component_ids.add(component_id)
        else:
            component_ids.add(-1)

        #assert(len(set(component_ids)) == 1)
        return list(component_ids)[0]

    def get_matched_label(self, mv):
        return self.matching_graph.get_vertex_property("matched_label", mv)

    def get_edge_conflicts(self):
        edge_conflicts = []
        for v in self.get_vertex_iterator():
            nbs = self.get_nbs(v, "matching")
            if self.is_groundtruth_mv(v):
                incident_edges = [[e.source(), e.target()] for e in self.matching_graph.get_incident_edges(v)]
            else:
                incident_edges = [[e.target(), e.source()] for e in self.matching_graph.get_incident_edges(v)]

            for i in range(len(nbs)):
                for j in range(i + 1, len(nbs)):
                    v_i = nbs[i]
                    v_j = nbs[j]
                    if self.get_label(v_i) != self.get_label(v_j):
                        edge_conflict = []
                        edge_conflict.extend([e for e in incident_edges if v_i in e or v_j in e])
                        edge_conflicts.append(edge_conflict)

        return edge_conflicts

    def get_edge_pairs(self):
        edge_pairs = set()
        for v0 in self.get_vertex_iterator():
            is_gt = self.is_groundtruth_mv(v0)
            v0_nbs_skeleton = self.get_nbs(v0, "skeleton")
            v0_nbs_matching = self.get_nbs(v0, "matching")
            for v1 in v0_nbs_skeleton:
                v1_nbs_matching = self.get_nbs(v1, "matching")
                
                for v0_nb in list(v0_nbs_matching) + [self.dummy_vertex]:
                    # Order vertices by gt vs rec, gt first
                    v0_nb_label = self.get_label(v0_nb)
                    if is_gt:
                        edge_0 = (int(v0), int(v0_nb))
                    else:
                        edge_0 = (int(v0_nb), int(v0))

                    for v1_nb in list(v1_nbs_matching) + [self.dummy_vertex]:
                        # We only look at labels of the neighbours
                        # as v0 and v1 have same label.
                        v1_nb_label = self.get_label(v1_nb)

                        if is_gt:
                            edge_1 = (int(v1), int(v1_nb))
                        else:
                            edge_1 = (int(v1_nb), int(v1))

                        # Is there an id switch?
                        switch = int(v0_nb_label != v1_nb_label)

                        # Order edges by gt id if possible
                        # if both are dummy order by rec
                        # consistent ordering assures no duplicates
                        # in the edge pairs
                        if edge_0[0] == edge_1[0] == (-1):
                            id0 = edge_0[1]
                            id1 = edge_1[1]
                        else:
                            id0 = edge_0[0]
                            id1 = edge_1[0]
                            
                        if id0 < id1:
                            edge_pairs.add(tuple([edge_0, edge_1, switch]))
                        else:
                            edge_pairs.add(tuple([edge_1, edge_0, switch]))

        return edge_pairs


    def __add_matched_labels(self):
        self.clear_edge_masks()
        self.clear_vertex_masks()

        self.matching_graph.new_vertex_property("matched_label", dtype="int", value=-1)

        for mv in self.matching_graph.get_vertex_iterator():
            nbs = self.get_nbs(mv, "matched")
            matched_label = self.__get_matched_label(nbs)
            self.matching_graph.set_vertex_property("matched_label", mv, matched_label)

    def __mark_splitmerge_edges(self):
        self.clear_edge_masks()
        self.clear_vertex_masks()

        self.matching_graph.new_edge_property("split", dtype="bool", value=False)
        self.matching_graph.new_edge_property("merge", dtype="bool", value=False)
        self.matching_graph.new_vertex_property("split", dtype="bool", value=False)
        self.matching_graph.new_vertex_property("merge", dtype="bool", value=False)

        self.mask_skeleton_edges()
        for e in self.matching_graph.get_edge_iterator():
            mv0 = e.source()
            mv1 = e.target()
            if self.get_matched_label(mv0) != self.get_matched_label(mv1):
                if self.is_groundtruth_mv(mv0):
                    assert(self.is_groundtruth_mv(mv1))
                    self.matching_graph.set_edge_property("split", e.source(), e.target(), True)
                    self.matching_graph.set_vertex_property("split", e.source(), True)
                    self.matching_graph.set_vertex_property("split", e.target(), True)
                else:
                    self.matching_graph.set_edge_property("merge", e.source(), e.target(), True)
                    self.matching_graph.set_vertex_property("merge", e.source(), True)
                    self.matching_graph.set_vertex_property("merge", e.target(), True)

        self.clear_edge_masks()

    def clear_edge_masks(self):
        self.matching_graph.set_edge_filter(None)

    def clear_vertex_masks(self):
        self.matching_graph.set_vertex_filter(None)

    def export_to_comatch(self, edge_conflicts=False, edge_pairs=False):
        logger.info("Export to comatch...")

        nodes_gt = []
        nodes_rec = []
        edges_gt_rec = []
        edge_costs = []
        labels_gt = {}
        labels_rec = {}

        for v in self.matching_graph.get_vertex_iterator():
            if self.is_groundtruth_mv(v):
                nodes_gt.append(int(v))
                labels_gt[int(v)] = self.get_label(v)
            else:
                nodes_rec.append(int(v))
                labels_rec[int(v)] = self.get_label(v)

        self.mask_matching_edges()
        for e in self.matching_graph.get_edge_iterator():
            edge_gt_rec = []
            if self.is_groundtruth_mv(e.source()):
                edge_gt_rec = [int(e.source()), int(e.target())]
            else:
                edge_gt_rec = [int(e.target()), int(e.source())]

            edges_gt_rec.append(tuple(edge_gt_rec))
            if self.distance_cost:
                edge_costs.append(self.get_distance(e))


        self.clear_edge_masks()

        if edge_conflicts:
            logger.info("Get edge conflicts...")
            edge_conflicts = self.get_edge_conflicts()
        else:
            edge_conflicts = None

        if edge_pairs:
            logger.info("Get edge pairs...")
            edge_pairs = self.get_edge_pairs()
        else:
            edge_pairs = None

        return nodes_gt, nodes_rec, edges_gt_rec, labels_gt, labels_rec, edge_costs, edge_conflicts, edge_pairs

    def import_matches(self, matches):
        self.clear_edge_masks()
        self.clear_vertex_masks()

        matched = self.matching_graph.new_edge_property("matched", dtype="bool", value=False)
        matched_vp = self.matching_graph.new_vertex_property("matched", dtype="bool", value=False)

        for match in matches:
            self.matching_graph.set_edge_property("matched", match[0], match[1], True)
            self.matching_graph.set_vertex_property("matched", match[0], True)
            self.matching_graph.set_vertex_property("matched", match[1], True)

        self.matched = True
        self.__mark_truefalse_vertices()
        self.__add_matched_labels()
        self.__mark_splitmerge_edges()

    def mask_matched_vertices(self):
        self.__mask_vp("matched")

    def mask_matched_edges(self):
        self.__mask_ep("matched")

    def mask_tps_rec(self):
        self.clear_edge_masks()
        self.clear_vertex_masks()

        is_tp = self.matching_graph.get_vertex_property("tp").a
        is_rec = np.logical_not(self.matching_graph.get_vertex_property("is_gt").a)

        tp_rec_vp = self.matching_graph.new_vertex_property("tp_rec", dtype="bool",  value=np.logical_and(is_tp, is_rec))
        self.matching_graph.set_vertex_filter(tp_rec_vp)

        self.mask_skeleton_edges()
 
    def export_tps_rec(self, path):
        self.mask_tps_rec()
        self.to_nml(path)

    def mask_tps_gt(self):
        self.clear_edge_masks()
        self.clear_vertex_masks()
       
        is_tp = self.matching_graph.get_vertex_property("tp").a
        is_gt = self.matching_graph.get_vertex_property("is_gt").a

        tp_gt_vp = self.matching_graph.new_vertex_property("tp_gt", dtype="bool",  value=np.logical_and(is_tp, is_gt))
        self.matching_graph.set_vertex_filter(tp_gt_vp)

        self.mask_skeleton_edges()

    def export_tps_gt(self, path):
        self.mask_tps_gt()
        self.to_nml(path)

    def export_fps(self, path):
        self.clear_edge_masks()
        self.clear_vertex_masks()

        self.mask_fps()
        self.mask_skeleton_edges()
        self.to_nml(path)

    def export_fns(self, path):
        self.clear_edge_masks()
        self.clear_vertex_masks()
        
        self.mask_fns()
        self.mask_skeleton_edges()
        self.to_nml(path)

    def export_splits(self, path):
        self.clear_edge_masks()
        self.clear_vertex_masks()

        self.mask_splits()
        self.to_nml(path)

    def export_mergers(self, path):
        self.clear_edge_masks()
        self.clear_vertex_masks()

        self.mask_mergers()
        self.to_nml(path)

    def export_matching(self, path):
        self.clear_edge_masks()
        self.clear_vertex_masks()

        self.mask_matched_edges()
        self.mask_matched_vertices()
        self.to_nml(path)

    def export_gt(self, path):
        self.clear_edge_masks()
        self.clear_vertex_masks()

        self.mask_groundtruth()
        self.to_nml(path)

    def export_rec(self, path):
        self.clear_edge_masks()
        self.clear_vertex_masks()

        self.mask_reconstruction()
        self.to_nml(path)

    def export_all(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        self.export_tps_rec(directory + "/tps_rec.nml")
        self.export_tps_gt(directory + "/tps_gt.nml")
        self.export_fps(directory + "/fps.nml")
        self.export_fns(directory + "/fns.nml")
        self.export_splits(directory + "/splits.nml")
        self.export_mergers(directory + "/mergers.nml")
        self.export_matching(directory + "/matching.nml")
        self.export_gt(directory + "/gt.nml")
        self.export_rec(directory + "/rec.nml")
        self.clear_vertex_masks()
        self.clear_edge_masks()
        self.to_nml(directory + "/input.nml")

        with open(directory + "/node_stats.txt", "w+") as f:
            json.dump(self.evaluate(), f)

    def evaluate(self):
        """
        Gathers error statistics on voxel level.
        tps, fps and fns are the number of matched/unmatched
        voxels while merges and splits are the number of 
        edges that connect to vertices with a different
        matching id.
        """

        stats = {"vertices": None,
                 "edges": None,
                 "tps_rec": None,
                 "tps_gt": None,
                 "fps": None,
                 "fns": None,
                 "merges": None,
                 "splits": None}

        self.clear_edge_masks()
        self.clear_vertex_masks()
        stats["vertices"] = self.get_number_of_vertices()

        self.mask_skeleton_edges()
        stats["edges"] = self.get_number_of_edges()
        
        self.mask_tps_rec()
        stats["tps_rec"] = self.get_number_of_vertices()

        self.mask_tps_gt()
        stats["tps_gt"] = self.get_number_of_vertices()

        self.mask_fps()
        stats["fps"] = self.get_number_of_vertices()

        self.mask_fns()
        stats["fns"] = self.get_number_of_vertices()

        self.mask_mergers()
        stats["merges"] = self.get_number_of_edges()

        self.mask_splits()
        stats["splits"] = self.get_number_of_edges()

        return stats
        

    def to_nml(self, path):
        g1_to_nml(self.matching_graph, path, voxel_size=self.voxel_size, knossify=True)

    def __add_skeleton_edges(self):
        """
        Add original skeleton edges back to the matching graph.
        """
        logger.info("Add skeleton edges...")

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

        logger.info("Add matching edges...")

        gt_positions = self.get_positions("gt")
        rec_positions = self.get_positions("rec")

        # tag_positions[i] == matching_graph.get_position[tag_mv_ids[i]]
        gt_mv_ids = self.get_mv_ids("gt")
        rec_mv_ids = self.get_mv_ids("rec")

        logger.info("Initialize KDTrees...")
        gt_tree = KDTree(gt_positions * np.array(voxel_size))
        rec_tree = KDTree(rec_positions * np.array(voxel_size))

        """
        From the docs:
        KDTree.query_ball_tree(other, r, p=2.0, eps=0)
        For each element self.data[i] of this tree, 
        results[i] is a list of the indices of its neighbors in other.data.
        """
        logger.info("Query ball tree...")
        results = gt_tree.query_ball_tree(rec_tree, r=distance_threshold)
       
        logger.info("Add matching edges to graph...")

        for gt_id in range(len(results)):
            mv_id_source = gt_mv_ids[gt_id]

            for rec_id in results[gt_id]:
                mv_id_target = rec_mv_ids[rec_id]
                edge = self.add_edge(mv_id_source, mv_id_target)
                # Set matching edge property:
                self.set_matching(edge)
                # Add distance:
                self.set_distance(edge)
