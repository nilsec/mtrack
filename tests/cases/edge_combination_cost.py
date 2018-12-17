import unittest
import numpy as np
from mtrack.graphs import g1_graph

class SmallCircleGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.g1_vertices_N = 10
        self.g1 = g1_graph.G1(self.g1_vertices_N)

class SmallSquareGraphTestCase(unittest.TestCase):
        # orientation = (1, 0, 0)
        # --->

        #  1----2
        #  |    |
        #  |    |
        #  0----3
 
    def setUp(self):
        self.g1_vertices_N = 4
        
        self.g1 = g1_graph.G1(self.g1_vertices_N)
        
        self.g1.set_position(0, np.array([0.0, 0.0, 0.0]))
        self.g1.set_position(1, np.array([0.0, 1.0, 0.0]))
        self.g1.set_position(2, np.array([1.0, 1.0, 0.0]))
        self.g1.set_position(3, np.array([1.0, 0.0, 0.0]))

        self.g1.add_edge(0, 1)
        self.g1.add_edge(1, 2)
        self.g1.add_edge(2, 3)
        self.g1.add_edge(3, 0)

        self.orientation = np.array([1.0, 0.0, 0.0])

        for v in self.g1.get_vertex_iterator():
            self.g1.set_orientation(v, self.orientation)

class IncreasingEnergyGraphTestCase(unittest.TestCase):
        # orientation = (1, 0, 0)
        # --->

        #  1----2
        #  |    
        #  |    
        #  0
 
        # vs:
        #
        # 1 
        # | \
        # |  \
        # 0   2

    def setUp(self):
        self.g1_vertices_N = 3
        
        self.g1 = g1_graph.G1(self.g1_vertices_N)
        self.g1.set_position(0, np.array([0.0, 0.0, 0.0]))
        self.g1.set_position(1, np.array([0.0, 1.0, 0.0]))
        self.g1.set_position(2, np.array([1.0, 1.0, 0.0]))
        self.g1.add_edge(0, 1)
        self.g1.add_edge(1, 2)

        self.orientation = np.array([1.0, 0.0, 0.0])

        for v in self.g1.get_vertex_iterator():
            self.g1.set_orientation(v, self.orientation)

        self.g1_high = g1_graph.G1(self.g1_vertices_N)
        self.g1_high.set_position(0, np.array([0.0, 0.0, 0.0]))
        self.g1_high.set_position(1, np.array([0.0, 1.0, 0.0]))
        self.g1_high.set_position(2, np.array([1.0, 0.0, 0.0]))
        self.g1_high.add_edge(0, 1)
        self.g1_high.add_edge(1, 2)

class CompareEdgeCombinationCostTestCase(IncreasingEnergyGraphTestCase):
    def runTest(self):
        comb_angle_factor = 1.0

        edge_combination_cost_low = self.g1.get_edge_combination_cost(comb_angle_factor)
        edge_combination_cost_high = self.g1_high.get_edge_combination_cost(comb_angle_factor)

        e_01 = self.g1.get_edge(0,1)
        e_12 = self.g1.get_edge(1,2)
        start_edge = self.g1.START_EDGE

        e_01_high = self.g1_high.get_edge(0,1)
        e_12_high = self.g1_high.get_edge(1,2)
        start_edge_high = self.g1_high.START_EDGE

        self.assertTrue(edge_combination_cost_low[(e_01, e_12)] < edge_combination_cost_high[(e_01_high,e_12_high)])


class GetEdgeCombinationCostTestCase(SmallSquareGraphTestCase):
    def runTest(self):
        comb_angle_factor =  1.0

        edge_combination_cost = self.g1.get_edge_combination_cost(comb_angle_factor)

        e_01 = self.g1.get_edge(0,1)
        e_12 = self.g1.get_edge(1,2)
        e_23 = self.g1.get_edge(2,3)
        e_30 = self.g1.get_edge(3,0)
        start_edge = self.g1.START_EDGE

        # Possible Combinations:
        #   1. Each edge with start edge: 4
        #   2.      e_01, e_12
        #           e_12, e_23
        #           e_23, e_30
        #           e_30, e_01
        #
        #   --> 8 edge pairs
        self.assertEqual(len(edge_combination_cost), 8)

        self.assertEqual(edge_combination_cost[(e_01, e_12)], edge_combination_cost[(e_12, e_23)])
        self.assertEqual(edge_combination_cost[(e_12, e_23)], edge_combination_cost[(e_23, e_30)])
        self.assertEqual(edge_combination_cost[(e_23, e_30)], edge_combination_cost[(e_01, e_30)])
 
        self.assertEqual(edge_combination_cost[(e_01, start_edge)], 0.0)
        self.assertEqual(edge_combination_cost[(e_12, start_edge)], 0.0)
        self.assertEqual(edge_combination_cost[(e_23, start_edge)], 0.0)
        self.assertEqual(edge_combination_cost[(e_30, start_edge)], 0.0)


if __name__ == "__main__":
    unittest.main()
