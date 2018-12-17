import unittest
import numpy as np
from mtrack.graphs import cost_converter, graph_converter, g1_graph, g2_solver
from mtrack import g2_to_g1_solution

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

        self.vertex_cost_params = {}
        self.edge_cost_params = {"orientation_factor": 10.0,
                                 "start_edge_prior": 0.0}
        self.edge_combination_cost_params = {"comb_angle_factor": 1.0}

        self.selection_cost = -100.0


        c_converter = cost_converter.CostConverter(self.g1,
                                                   self.vertex_cost_params,
                                                   self.edge_cost_params,
                                                   self.edge_combination_cost_params,
                                                   self.selection_cost)


        g2_cost = c_converter.get_g2_cost(self.g2, 
                                          self.index_maps)

        for v in self.g2.get_vertex_iterator():
            self.g2.set_cost(v, g2_cost[v])


class SolverTestCase(SmallSquareGraphTestCase):
    def runTest(self):
        solver = g2_solver.G2Solver(self.g2)
        solution = solver.solve()
        g1_solution = g2_to_g1_solution(solution,
                                        self.g1,
                                        self.g2,
                                        self.index_maps)

        self.assertEqual(g1_solution.get_number_of_edges(), 2)
        self.assertEqual(g1_solution.get_number_of_vertices(), 4)

        for e in g1_solution.get_edge_iterator():
            v0 = e.source()
            v1 = e.target()
            diff = np.abs(np.array(g1_solution.get_position(v0)) -\
                          np.array(g1_solution.get_position(v1)))
            self.assertTrue(np.all(diff == self.orientation))

if __name__ == "__main__":
    unittest.main()
