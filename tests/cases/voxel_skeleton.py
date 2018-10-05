import unittest
import numpy as np

from mtrack.graphs import G1
from mtrack.evaluation.voxel_skeleton import VoxelSkeleton
from mtrack.preprocessing import g1_to_nml
import pdb


class VoxelSkeletonBaseCase(unittest.TestCase):
    def setUp(self):
        self.vertices = 20
        self.g1 = G1(self.vertices)
        self.points = []

        x = 100
        y = 100
        z = 10

        for v in self.g1.get_vertex_iterator():
            self.g1.set_position(v, np.array([x,y,z]))
            self.g1.set_orientation(v, np.array([1.,0.,0.]))
            self.points.append(np.array([x,y,z]))

            x += 5
            y += 10
            z += 15

            if int(v) < self.vertices - 1:
                self.g1.add_edge(int(v), int(v) + 1)


class VoxelSkeletonTestCase(VoxelSkeletonBaseCase):
    def runTest(self):
        vs = VoxelSkeleton(self.g1, voxel_size=[1.,1.,1.], verbose=True)
        voxel_graph = vs.get_graph()
        voxel_points = vs.get_points()

        voxel_graph_points = []
        for v in voxel_graph.get_vertex_iterator():
            voxel_graph_points.append(np.array(voxel_graph.get_position(v)))

        for point in voxel_points:
            self.assertTrue(point in np.array(voxel_graph_points))

        for point in self.points:
            self.assertTrue(point in np.array(voxel_graph_points))

if __name__ == "__main__":
    unittest.main()
