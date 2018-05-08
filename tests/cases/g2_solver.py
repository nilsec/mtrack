import unittest
import numpy as np
from mtrack.graphs import cost_converter, graph_converter, g1_graph, g2_solver


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

        vertex_cost_params = {}
        edge_cost_params = {"distance_factor": 1.0, 
                            "orientation_factor": 10.0,
                            "start_edge_prior": 0.0}
        edge_combination_cost_params = {"comb_angle_factor": 1.0}

        selection_cost = -100.0


        c_converter = cost_converter.CostConverter(self.g1,
                                                   vertex_cost_params,
                                                   edge_cost_params,
                                                   edge_combination_cost_params,
                                                   selection_cost)


        g2_cost = c_converter.get_g2_cost(self.g2, 
                                          self.index_maps)

        for v in self.g2.get_vertex_iterator():
            self.g2.set_cost(v, g2_cost[v])


class SolverTestCase(SmallSquareGraphTestCase):
    def runTest(self):
        solver = g2_solver.G2Solver(self.g2)
        g_solution = solver.solve()

        print "g_solution: \n"
        for i in range(len(g_solution)):
            print "y_" + str(i) + ": " + str(g_solution[i])

if __name__ == "__main__":
    unittest.main()
