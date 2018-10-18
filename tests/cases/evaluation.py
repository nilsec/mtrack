import unittest
import numpy as np

from mtrack.graphs import G1
from mtrack.evaluation.matching_graph import MatchingGraph
from mtrack.evaluation.voxel_skeleton import VoxelSkeleton
from mtrack.evaluation.evaluate import build_matching_graph, evaluate_matching_graph
from comatch import match_components
import json

test_data_dir = "./data"

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

if __name__ == "__main__":
    unittest.main()
