import unittest
import numpy as np

from mtrack.graphs import G1
from mtrack.evaluation.matching_graph import MatchingGraph
from mtrack.evaluation.voxel_skeleton import VoxelSkeleton
from comatch import match_components
import json

test_data_dir = "./data"

class ParallelLinesSetUp(unittest.TestCase):
    def setUp(self):
        """

        o       o
        |       |
        |       |
        |       |
        |       |
        |       |
        o       o
        |       |
        |       |
        |       |
        |       |
        o       o
        |       |
        |       |
        .       .
        .       .
        .       .

        """


        self.gt_vertices = 10
        self.rec_vertices = 10

        self.gt = G1(self.gt_vertices)
        self.rec = G1(self.rec_vertices)

        z = 0
        for v in self.gt.get_vertex_iterator():
            self.gt.set_position(v, np.array([100,100,z]))
            self.gt.set_orientation(v, np.array([1,0,0]))

            z += 5
            if int(v)<self.gt_vertices-1:
                self.gt.add_edge(int(v), int(v)+1)

        self.vs_gt = VoxelSkeleton(self.gt, voxel_size=[1.,1.,1.], verbose=True)


        # Different offset:
        z = 0
        for v in self.rec.get_vertex_iterator():
            self.rec.set_position(v, np.array([150,100,z]))
            self.rec.set_orientation(v, np.array([1,0,0]))

            z += 5
            if int(v)<self.rec_vertices-1:
                self.rec.add_edge(int(v), int(v)+1)

        self.vs_rec = VoxelSkeleton(self.rec, voxel_size=[1.,1.,1.], verbose=True)

        self.groundtruth_skeletons = [self.vs_gt]
        self.reconstructed_skeletons = [self.vs_rec]

        self.skeletons = {"gt": self.vs_gt, "rec": self.vs_rec}

        self.distance_threshold = 50.1
        self.voxel_size = [1.,1.,1.]


class ErrorTestSetUpSameDistance(unittest.TestCase):
    def setUp(self):
        self.gt_vertices = 10
        self.rec1_vertices = 5
        self.rec2_vertices = 5

        self.gt = G1(self.gt_vertices)
        self.rec1 = G1(self.rec1_vertices)
        self.rec2 = G1(self.rec2_vertices)

        z = 0
        for v in self.gt.get_vertex_iterator():
            self.gt.set_position(v, np.array([100,100,z]))
            self.gt.set_orientation(v, np.array([1,0,0]))

            z += 5
            if int(v)<self.gt_vertices-1:
                self.gt.add_edge(int(v), int(v)+1)

        self.vs_gt = VoxelSkeleton(self.gt, voxel_size=[1.,1.,1.], verbose=True, subsample=5)


        z = 0
        for v in self.rec1.get_vertex_iterator():
            self.rec1.set_position(v, np.array([160,100, z]))
            self.rec1.set_orientation(v, np.array([1,0,0]))

            z += 5
            if int(v)<self.rec1_vertices-1:
                self.rec1.add_edge(int(v), int(v)+1)


        z = 0
        for v in self.rec2.get_vertex_iterator():
            self.rec2.set_position(v, np.array([100,150, z]))
            self.rec2.set_orientation(v, np.array([1,0,0]))

            z += 5
            if int(v)<self.rec2_vertices-1:
                self.rec2.add_edge(int(v), int(v)+1)

        self.vs_rec1 = VoxelSkeleton(self.rec1, voxel_size=[1.,1.,1.], verbose=True, subsample=5)
        self.vs_rec2 = VoxelSkeleton(self.rec2, voxel_size=[1.,1.,1.], verbose=True, subsample=5)

        self.groundtruth_skeletons = [self.vs_gt]
        self.reconstructed_skeletons = [self.vs_rec1, self.vs_rec2]

        self.skeletons = {"gt": [self.vs_gt], "rec": [self.vs_rec1, self.vs_rec2]}

        self.distance_threshold = 51
        self.voxel_size = [1.,1.,1.]


class MatchingGraphNoInitAllTestCase(ParallelLinesSetUp):
    def runTest(self):
        mg = MatchingGraph(self.groundtruth_skeletons, 
                           self.reconstructed_skeletons,
                           self.distance_threshold,
                           self.voxel_size,
                           verbose=True,
                           initialize_all=False)

        self.assertTrue(mg.total_vertices ==\
                        self.vs_gt.get_graph().get_number_of_vertices() +\
                        self.vs_rec.get_graph().get_number_of_vertices())

class MatchingGraphInitializeTestCase(ParallelLinesSetUp):
    def runTest(self):
        mg = MatchingGraph(self.groundtruth_skeletons, 
                           self.reconstructed_skeletons,
                           self.distance_threshold,
                           self.voxel_size,
                           verbose=True,
                           initialize_all=False)

        # Test private methods too as internals are complex:
        matching_graph, mappings, mv_to_v, v_to_mv = mg._MatchingGraph__initialize()
        
        self.assertTrue(matching_graph.get_number_of_vertices() ==\
                        mg._MatchingGraph__get_total_vertices())

        self.assertTrue(matching_graph.get_number_of_edges()==0)

        for tag in ["gt", "rec"]:
            for graph in mg.graphs[tag]:
                for v in graph.get_vertex_iterator():
                    mv = v_to_mv[(graph, int(v))]
                    
                    pos_v = np.array(graph.get_position(v))
                    pos_mv = np.array(matching_graph.get_position(mv))
                    self.assertTrue(np.all(pos_v == pos_mv))


        mv_ids_rec = mappings["rec"]["mv_ids"]
        mv_ids_gt = mappings["gt"]["mv_ids"]
        self.assertTrue(set(mv_ids_rec) & set(mv_ids_gt) == set([]))
        self.assertTrue(sorted(mv_ids_rec + mv_ids_gt) ==\
                        range(matching_graph.get_number_of_vertices()))

        
        for i in range(len(mv_ids_gt)):
            mv_id = mv_ids_gt[i]
            graph_pos = np.array(matching_graph.get_position(mv_id))
            mapping_pos = mappings["gt"]["positions"][i]
            self.assertTrue(np.all(graph_pos == mapping_pos))

        for i in range(len(mv_ids_rec)):
            mv_id = mv_ids_rec[i]
            graph_pos = np.array(matching_graph.get_position(mv_id))
            mapping_pos = mappings["rec"]["positions"][i]
            self.assertTrue(np.all(graph_pos == mapping_pos))

class MatchingGraphAddSkeletonEdgesTestCase(ParallelLinesSetUp):
    def runTest(self):
        mg = MatchingGraph(self.groundtruth_skeletons, 
                           self.reconstructed_skeletons,
                           self.distance_threshold,
                           self.voxel_size,
                           verbose=True,
                           initialize_all=False)

        matching_graph, mappings, mv_to_v, v_to_mv = mg._MatchingGraph__initialize()
        mg.matching_graph = matching_graph
        mg.mappings = mappings
        mg.mv_to_v = mv_to_v
        mg.v_to_mv = v_to_mv

        self.assertTrue(matching_graph.get_number_of_edges() == 0)
        mg._MatchingGraph__add_skeleton_edges()

        self.assertTrue(matching_graph.get_number_of_edges() ==\
                        self.vs_gt.get_graph().get_number_of_edges() +\
                        self.vs_rec.get_graph().get_number_of_edges())

        # Check that all edges are attached to the correct vertices:
        for e in matching_graph.get_edge_iterator():
            mv0 = e.source()
            mv1 = e.target()

            v0 = mv_to_v[mv0]
            v1 = mv_to_v[mv1]

            # Compare graphs
            self.assertTrue(v0[0] == v1[0])
            edge = v0[0].get_edge(v0[1], v1[1]) # Raises value error if not there

            pos_v0 = np.array(v0[0].get_position(v0[1]))
            pos_v1 = np.array(v0[0].get_position(v1[1]))
            
            pos_mv0 = np.array(matching_graph.get_position(mv0))
            pos_mv1 = np.array(matching_graph.get_position(mv1))

            self.assertTrue(np.all(pos_v0 == pos_mv0))
            self.assertTrue(np.all(pos_v1 == pos_mv1))

class MatchingGraphAddMatchingEdgesTestCase(ParallelLinesSetUp):
    def runTest(self):
        mg = MatchingGraph(self.groundtruth_skeletons, 
                           self.reconstructed_skeletons,
                           self.distance_threshold,
                           self.voxel_size,
                           verbose=True,
                           initialize_all=False)

        matching_graph, mappings, mv_to_v, v_to_mv = mg._MatchingGraph__initialize()
        mg.matching_graph = matching_graph
        mg.mappings = mappings
        mg.mv_to_v = mv_to_v
        mg.v_to_mv = v_to_mv

        mg._MatchingGraph__add_skeleton_edges()
        edges_pre_add = mg.matching_graph.get_number_of_edges()

        mg._MatchingGraph__add_matching_edges(self.distance_threshold, self.voxel_size)
        edges_post_add = mg.matching_graph.get_number_of_edges()

        self.assertTrue(edges_post_add > edges_pre_add)
        mg.mask_skeleton_edges()
        edges_post_masking = mg.matching_graph.get_number_of_edges()
        self.assertTrue(edges_post_masking == edges_pre_add)
        
        for e in mg.matching_graph.get_edge_iterator():
            self.assertTrue(mg.get_edge_type(e) == "skeleton")

        mg.clear_edge_masks()
        self.assertTrue(edges_post_add == mg.matching_graph.get_number_of_edges())

        mg.mask_matching_edges()
        self.assertTrue(edges_post_add - edges_pre_add ==\
                        mg.matching_graph.get_number_of_edges())

        for e in mg.matching_graph.get_edge_iterator():
            self.assertTrue(mg.get_edge_type(e) == "matching")
            
            v0_gt = mg.is_groundtruth_mv(e.source())
            v1_gt = mg.is_groundtruth_mv(e.target())
            self.assertTrue(int(v0_gt) != int(v1_gt))

        mg.clear_edge_masks()
        mg.to_nml(test_data_dir + "/matching_graph.nml")

class MatchingGraphExportToComatch(ParallelLinesSetUp):
    def runTest(self):
        mg = MatchingGraph(self.groundtruth_skeletons, 
                           self.reconstructed_skeletons,
                           self.distance_threshold,
                           self.voxel_size,
                           verbose=True,
                           initialize_all=True)
        
        nodes_gt, nodes_rec, edges_gt_rec, labels_gt, labels_rec, edge_costs, edge_conflicts, edge_pairs = mg.export_to_comatch()

        for v_gt in nodes_gt:
            self.assertTrue(mg.is_groundtruth_mv(v_gt))

        for v_rec in nodes_rec:
            self.assertFalse(mg.is_groundtruth_mv(v_rec))

        mg.mask_matching_edges()
        self.assertTrue(len(edges_gt_rec) == mg.get_number_of_edges())

        mg.clear_edge_masks()
        self.assertTrue(len(nodes_gt) + len(nodes_rec) == mg.get_number_of_vertices())

class MatchingGraphImportMatches(ParallelLinesSetUp):
    def runTest(self):
        print "Import matches"

        mg = MatchingGraph(self.groundtruth_skeletons, 
                           self.reconstructed_skeletons,
                           self.distance_threshold,
                           self.voxel_size,
                           verbose=True,
                           initialize_all=True)

        nodes_gt, nodes_rec, edges_gt_rec, labels_gt, labels_rec, edge_costs, edge_conflicts, edge_pairs = mg.export_to_comatch()

        label_matches, node_matches, num_splits, num_merges, num_fps, num_fns = match_components(nodes_gt, nodes_rec, edges_gt_rec, labels_gt, labels_rec)
        matches = node_matches

        # Everything is matched
        self.assertTrue(len(matches) == mg.get_number_of_vertices()/2)
        mg.import_matches(matches)

        for v in mg.get_vertex_iterator():
            self.assertTrue(mg.is_tp(v))
            self.assertFalse(mg.is_fp(v))
            self.assertFalse(mg.is_fn(v))

        for e in mg.get_edge_iterator():
            self.assertFalse(mg.is_split(e))
            self.assertFalse(mg.is_merge(e))


class MatchingGraphExportToComatch(ParallelLinesSetUp):
    def runTest(self):
        mg = MatchingGraph(self.groundtruth_skeletons, 
                           self.reconstructed_skeletons,
                           self.distance_threshold,
                           self.voxel_size,
                           verbose=True,
                           initialize_all=True)
        
        nodes_gt, nodes_rec, edges_gt_rec, labels_gt, labels_rec, edge_costs, edge_conflicts, edge_pairs = mg.export_to_comatch()

        for v_gt in nodes_gt:
            self.assertTrue(mg.is_groundtruth_mv(v_gt))

        for v_rec in nodes_rec:
            self.assertFalse(mg.is_groundtruth_mv(v_rec))

        mg.mask_matching_edges()
        self.assertTrue(len(edges_gt_rec) == mg.get_number_of_edges())

        mg.clear_edge_masks()
        self.assertTrue(len(nodes_gt) + len(nodes_rec) == mg.get_number_of_vertices())

class TestOneToOne(ErrorTestSetUpSameDistance):
    def runTest(self):
        print "OneToOne"

        mg = MatchingGraph(self.groundtruth_skeletons, 
                           self.reconstructed_skeletons,
                           self.distance_threshold,
                           self.voxel_size,
                           verbose=True,
                           distance_cost=True,
                           initialize_all=True)

        nodes_gt, nodes_rec, edges_gt_rec, labels_gt, labels_rec, edge_costs, edge_conflicts, edge_pairs = mg.export_to_comatch()
        try:
            # Quadmatch
            label_matches, node_matches, num_splits, num_merges, num_fps, num_fns = match_components(nodes_gt, nodes_rec, edges_gt_rec, labels_gt, 
                                                                                                     labels_rec, edge_conflicts=edge_conflicts, max_edges=1, edge_costs=edge_costs)
        except TypeError:
            # Comatch
            label_matches, node_matches, num_splits, num_merges, num_fps, num_fns = match_components(nodes_gt, nodes_rec, edges_gt_rec, labels_gt, 
                                                                                                     labels_rec, allow_many_to_many=False, edge_costs=edge_costs, no_match_costs=1000.)
 
        print "label matches:", label_matches
        print "node_matches:", node_matches

        comatch_errors = {"splits": num_splits, "num_merges": num_merges, "num_fps": num_fps, "num_fns": num_fns}
        print comatch_errors

        mg.import_matches(node_matches)
        output_dir = test_data_dir + "/MatchingOnetoOne" 
        mg.export_all(output_dir)

        with open(output_dir + "/macro_errors.json", "w+") as f:
            json.dump(comatch_errors, f)

class TestManyToMany(ErrorTestSetUpSameDistance):
    def runTest(self):
        print "ManyToMany"

        mg = MatchingGraph(self.groundtruth_skeletons, 
                           self.reconstructed_skeletons,
                           self.distance_threshold,
                           self.voxel_size,
                           verbose=True,
                           distance_cost=True,
                           initialize_all=True)

        nodes_gt, nodes_rec, edges_gt_rec, labels_gt, labels_rec, edge_costs, edge_conflicts, edge_pairs = mg.export_to_comatch()
        try:
            # Quadmatch
            label_matches, node_matches, num_splits, num_merges, num_fps, num_fns = match_components(nodes_gt, nodes_rec, edges_gt_rec, labels_gt, 
                                                                                                     labels_rec, edge_conflicts=edge_conflicts, max_edges=10, edge_costs=edge_costs)
        except TypeError:
            # Comatch
            label_matches, node_matches, num_splits, num_merges, num_fps, num_fns = match_components(nodes_gt, nodes_rec, edges_gt_rec, labels_gt, 
                                                                                                     labels_rec, allow_many_to_many=True, edge_costs=edge_costs, no_match_costs=1000.)
 
        print "label matches:", label_matches
        print "node_matches:", node_matches

        comatch_errors = {"splits": num_splits, "num_merges": num_merges, "num_fps": num_fps, "num_fns": num_fns}
        print comatch_errors

        mg.import_matches(node_matches)
        output_dir = test_data_dir + "/MatchingManytoMany" 
        mg.export_all(output_dir)

        with open(output_dir + "/macro_errors.json", "w+") as f:
            json.dump(comatch_errors, f)


if __name__ == "__main__":
    unittest.main()
