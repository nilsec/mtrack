import unittest
import numpy as np

from mtrack.graphs import G1
from mtrack.evaluation.matching_graph import MatchingGraph
from mtrack.evaluation.voxel_skeleton import VoxelSkeleton
from mtrack.evaluation.evaluate import build_matching_graph, evaluate_matching_graph, evaluate
from comatch import match_components
import json
import pylp

test_data_dir = "/groups/funke/home/ecksteinn/Projects/microtubules/mtrack/tests/cases/data"

class SingleTrajectorySetUpWithScaling(unittest.TestCase):
    def setUp(self):
        self.gt_vertices = 10
        self.rec_vertices = 5
        self.distance_threshold = 61 * 5
        self.voxel_size = [5.,5.,50.]

        self.gt = G1(self.gt_vertices)
        self.rec = G1(self.rec_vertices)

        z = 0
        for v in self.gt.get_vertex_iterator():
            self.gt.set_position(v, np.array([100,100,z]))
            self.gt.set_orientation(v, np.array([1,0,0]))

            z += 5
            if int(v)<self.gt_vertices-1:
                self.gt.add_edge(int(v), int(v)+1)

        self.vs_gt = VoxelSkeleton(self.gt, voxel_size=self.voxel_size, verbose=True, subsample=5)


        z = 0
        for v in self.rec.get_vertex_iterator():
            self.rec.set_position(v, np.array([160,100, z]))
            self.rec.set_orientation(v, np.array([1,0,0]))

            z += 5
            if int(v)<self.rec_vertices-1:
                self.rec.add_edge(int(v), int(v)+1)

        self.vs_rec = VoxelSkeleton(self.rec, voxel_size=self.voxel_size, verbose=True, subsample=5)

        self.groundtruth_skeletons = [self.vs_gt]
        self.reconstructed_skeletons = [self.vs_rec]

class SingleTrajectorySetUpNoScaling(unittest.TestCase):
    def setUp(self):
        self.gt_vertices = 10
        self.rec_vertices = 5
        self.distance_threshold = 61 * 1
        self.voxel_size = [1.,1.,1.]

        self.gt = G1(self.gt_vertices)
        self.rec = G1(self.rec_vertices)

        z = 0
        for v in self.gt.get_vertex_iterator():
            self.gt.set_position(v, np.array([100,100,z]))
            self.gt.set_orientation(v, np.array([1,0,0]))

            z += 5
            if int(v)<self.gt_vertices-1:
                self.gt.add_edge(int(v), int(v)+1)

        self.vs_gt = VoxelSkeleton(self.gt, voxel_size=self.voxel_size, verbose=True, subsample=5)


        z = 0
        for v in self.rec.get_vertex_iterator():
            self.rec.set_position(v, np.array([160,100, z]))
            self.rec.set_orientation(v, np.array([1,0,0]))

            z += 5
            if int(v)<self.rec_vertices-1:
                self.rec.add_edge(int(v), int(v)+1)

        self.vs_rec = VoxelSkeleton(self.rec, voxel_size=self.voxel_size, verbose=True, subsample=5)

        self.groundtruth_skeletons = [self.vs_gt]
        self.reconstructed_skeletons = [self.vs_rec]

"""
class EvaluateMatchingGraphTestCaseScaling(SingleTrajectorySetUpWithScaling):
    def runTest(self):
        mg = build_matching_graph(self.gt, self.rec, self.voxel_size, distance_threshold=self.distance_threshold, subsample=5)
        mg, terr, nerr= evaluate_matching_graph(mg, export_to=test_data_dir + "/evaluate_test")
        print "Scaling: ", terr, nerr

class EvaluateMatchingGraphTestCase(SingleTrajectorySetUpNoScaling):
    def runTest(self):
        mg = build_matching_graph(self.gt, self.rec, self.voxel_size, distance_threshold=self.distance_threshold, subsample=5)
        mg, terr, nerr= evaluate_matching_graph(mg, export_to=test_data_dir + "/evaluate_test")
        print "No scaling:", terr, nerr
"""

class WrapperTestCase(unittest.TestCase):
    def runTest(self):
        tracing = test_data_dir + "/wrapper_test/master_300_329_v1.nml"
        reconstruction = test_data_dir + "/wrapper_test/validated.nml"
        x_lim = {"min": 12 + 100, "max": 1012 - 100}
        y_lim = {"min": 12 + 100, "max": 1012 - 100}
        z_lim = {"min": 300,"max": 330}

        pylp.set_log_level(pylp.LogLevel.Debug)

        matching_graph, topological_errors, node_errors = evaluate(tracing, 
                                                                   reconstruction, 
                                                                   voxel_size=[5.,5.,50.],
                                                                   distance_threshold=150,
                                                                   subsample=10,
                                                                   export_to=test_data_dir + "/wrapper_test/results_quadmatch",
                                                                   x_lim=x_lim,
                                                                   y_lim=y_lim,
                                                                   z_lim=z_lim,
                                                                   optimality_gap=0,
                                                                   use_distance_costs=False,
                                                                   absolute=True,
                                                                   time_limit=120)

        print "topological errors: ", topological_errors
        print "node errors: ", node_errors

if __name__ == "__main__":
    unittest.main()
