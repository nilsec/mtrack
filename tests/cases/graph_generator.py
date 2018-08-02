import unittest
import numpy as np

from mtrack.mt_utils.graph_generator import GraphGenerator

class BasicGraphGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        self.n_vertices = 100
        self.min_pos = 1
        self.max_pos = 101
        self.p_partner = 0.5
        self.generator = GraphGenerator(self.n_vertices,
                                        self.min_pos,
                                        self.max_pos,
                                        self.p_partner)

        self.assertEqual(self.generator.g1.get_number_of_vertices(),
                         self.n_vertices)

        for v in self.generator.g1.get_vertex_iterator():
            pos = np.array(self.generator.g1.get_position(v))
            self.assertTrue(np.all(pos<=np.array([self.max_pos] * 3)))
            self.assertTrue(np.all(pos>=np.array([self.min_pos] * 3)))


class RandomConnectivityTestCase(BasicGraphGeneratorTestCase):
    def runTest(self):
        self.n_edges = 500
        g1_random = self.generator.random_connectivity(self.n_edges)
        
        self.assertEqual(self.n_edges, g1_random.get_number_of_edges())
        

class LocalConnectivityTestCase(BasicGraphGeneratorTestCase):
    def runTest(self):
        self.distance_threshold = (self.max_pos - self.min_pos)/10.
        g1_random = self.generator.local_connectivity(self.distance_threshold)
        
        n_edges = 0
        for v0 in g1_random.get_vertex_iterator():
            p0 = np.array(g1_random.get_position(v0))
            partner0 = g1_random.get_partner(v0)
            for v1 in g1_random.get_vertex_iterator():
                if v1 > v0 and partner0 != v1:
                    p1 = np.array(g1_random.get_position(v1))
                    if np.linalg.norm(p1 - p0) <= self.distance_threshold:
                        n_edges += 1

        self.assertEqual(n_edges, g1_random.get_number_of_edges())

if __name__ == "__main__":
    unittest.main()

