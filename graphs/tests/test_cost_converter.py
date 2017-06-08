import sys
import os
sys.path.append(os.path.join('..', '..'))

import unittest
import numpy as np
from graphs import cost_converter, graph_converter, g1_graph


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
    
        self.graph_converter = graph_converter.GraphConverter(self.g1)
        self.g2, self.index_maps = self.graph_converter.get_g2_graph()
 

class GetG2CostTestCase(SmallSquareGraphTestCase):
    def runTest(self):
        vertex_cost_params = {}
        edge_cost_params = {"distance_factor": 1.0, 
                            "orientation_factor": 1.0,
                            "start_edge_prior": 20.0}
        edge_combination_cost_params = {"comb_angle_factor": 1.0}

        selection_cost = -1.0

        c_converter = cost_converter.CostConverter(self.g1,
                                                   vertex_cost_params,
                                                   edge_cost_params,
                                                   edge_combination_cost_params,
                                                   selection_cost)


        g2_cost = c_converter.get_g2_cost(self.g2, 
                                          self.index_maps)

        # Cost for non start edge containing g2 vertices:
        # comb_angle: caf * np.pi/2 ** 2
        # distance cost = df * (1.0 + 1.0) * 1/2 = 1.0
        # orientation_cost = of * (pi/2 + pi/2 + 0.0 + 0.0) * 0.5 = pi/2
        # selection_cost = selection_cost

        # Cost for start edge containing g2 vertices:
        # Start Edge prior: Start Edge Prior
        # distance cost: Same
        # Orientation cost: Same
        # Selection Cost: Same
        # Comb angle cost: 0.0 + (comb_angle_bias = 0.0)
        # (comb angle bias can be thought implicit in start edge prior)
        
        g2_v_nonstart_cost = edge_cost_params["orientation_factor"] * np.pi/2 +\
                             edge_cost_params["distance_factor"] * 1.0 +\
                             edge_combination_cost_params["comb_angle_factor"] * (np.pi/2)**2+\
                             selection_cost
 

        g2_v_start_cost_y = edge_cost_params["start_edge_prior"] +\
                          edge_cost_params["orientation_factor"] * np.pi/2 +\
                          edge_cost_params["distance_factor"] * 0.5 +\
                          selection_cost

        g2_v_start_cost_x = g2_v_start_cost_y - np.pi/2 * edge_cost_params["orientation_factor"]

        for v in g2_cost.keys():
            if g1_graph.G1.START_EDGE in self.index_maps["g2vertex_g1edges"][v]:
                if self.g1.get_edge(0,1) in self.index_maps["g2vertex_g1edges"][v] or\
                   self.g1.get_edge(2,3) in self.index_maps["g2vertex_g1edges"][v] or\
                   self.g1.get_edge(1,0) in self.index_maps["g2vertex_g1edges"][v] or\
                   self.g1.get_edge(3,2) in self.index_maps["g2vertex_g1edges"][v]:
                    #print v, g2_cost[v]
                    self.assertEqual(g2_cost[v], g2_v_start_cost_y)

                else:
                    #print self.index_maps["g2vertex_g1edges"][v]
                    #print v, g2_v_start_cost_x, g2_cost[v]
                    
                    self.assertEqual(g2_cost[v], g2_v_start_cost_x)

            else:
                self.assertEqual(g2_cost[v], g2_v_nonstart_cost)


if __name__ == "__main__":
    unittest.main()
