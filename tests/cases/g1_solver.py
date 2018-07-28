import unittest
import numpy as np
from mtrack.graphs import g1_graph, g1_solver, g2_solver, cost_converter, graph_converter
from g2_solver import SmallSquareGraphTestCase


class SolverTestCase(SmallSquareGraphTestCase):
    def runTest(self):
        solver = g1_solver.G1Solver(
            self.g1,
            distance_factor=self.edge_cost_params["distance_factor"],
            orientation_factor=self.edge_cost_params["orientation_factor"],
            start_edge_prior=self.edge_cost_params["start_edge_prior"],
            comb_angle_factor=self.edge_combination_cost_params["comb_angle_factor"],
            vertex_selection_cost=self.selection_cost,
            backend="Gurobi")

        solution = solver.solve()
        g1_solution = solver.solution_to_g1(solution,
                                            voxel_size=np.array([5.,5.,50.]))

        print "solution, n_edges: ", g1_solution.get_number_of_edges()
        print "solution, n_vertices: ", g1_solution.get_number_of_vertices()

if __name__ == "__main__":
    unittest.main()
