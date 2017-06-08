import sys
import os
sys.path.append(os.path.join('..', '..'))

import unittest
import numpy as np
from graphs import g1_graph

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

class SetGetOrientationTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        orientation = np.array([1.0, 0.11111, 0.123])
        
        for n in range(self.g1.get_number_of_vertices()):
            self.g1.set_orientation(n, orientation * n)

        for n in range(self.g1.get_number_of_vertices()):
            self.assertTrue(np.all(orientation * n == self.g1.get_orientation(n)))

class SetGetPositionTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        position = np.array([2.0, 340.0, 123])
        
        for n in range(self.g1.get_number_of_vertices()):
            self.g1.set_position(n, position * n)

        for n in range(self.g1.get_number_of_vertices()):
            self.assertTrue(np.all(position * n == self.g1.get_position(n)))

class SetGetPartnerTestCase(SmallCircleGraphTestCase):
    def runTest(self):
         
        for n in range(self.g1.get_number_of_vertices()):
            self.g1.set_partner(n, (n+1) % self.g1.get_number_of_vertices())

        for n in range(self.g1.get_number_of_vertices()):
            self.assertTrue(np.all((n+1) % self.g1.get_number_of_vertices() == self.g1.get_partner(n)))

class GetVertexCostTestCase(SmallSquareGraphTestCase):
    def runTest(self):
        vertex_cost = self.g1.get_vertex_cost()

        self.assertTrue(len(vertex_cost) == self.g1_vertices_N + 1) # + 1 for start edge
        
        for v in self.g1.get_vertex_iterator():
            self.assertTrue(vertex_cost[v] == 0)


class GetEdgeCostTestCase(SmallSquareGraphTestCase):
    def runTest(self):
        edge_cost_debug = self.g1.get_edge_cost(distance_factor=1.0, 
                                                orientation_factor=1.0,
                                                start_edge_prior=33.3,
                                                debug=True)
    
        for cost in edge_cost_debug.values():    
            self.assertEqual(cost["distance_cost"], 1.0)

        e_01 = self.g1.get_edge(0,1)
        self.assertEqual(edge_cost_debug[e_01]["orientation_cost"], np.pi) # pi/2 + pi/2 from vertex 0 & vertex 1
        
        e_12 = self.g1.get_edge(1,2)
        self.assertEqual(edge_cost_debug[e_12]["orientation_cost"], 0.0)

        e_23 = self.g1.get_edge(2,3)
        self.assertEqual(edge_cost_debug[e_23]["orientation_cost"], np.pi)

        e_30 = self.g1.get_edge(3,0)
        self.assertEqual(edge_cost_debug[e_30]["orientation_cost"], 0.0)


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

        self.assertEqual(edge_combination_cost[(e_01, e_12)], (np.pi/2.)**2)
        self.assertEqual(edge_combination_cost[(e_12, e_23)], (np.pi/2.)**2)
        self.assertEqual(edge_combination_cost[(e_23, e_30)], (np.pi/2.)**2)
        self.assertEqual(edge_combination_cost[(e_01, e_30)], (np.pi/2.)**2)
 
        self.assertEqual(edge_combination_cost[(e_01, start_edge)], 0.0)
        self.assertEqual(edge_combination_cost[(e_12, start_edge)], 0.0)
        self.assertEqual(edge_combination_cost[(e_23, start_edge)], 0.0)
        self.assertEqual(edge_combination_cost[(e_30, start_edge)], 0.0)
 
if __name__ == "__main__":
    unittest.main()
