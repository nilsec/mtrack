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

        dummys = []
        for d in solver.dummy_to_binary.values():
            if solution[d] > 0.5:
                dummys.append(1)

        vertices = []
        for v in solver.vertex_to_binary.values():
            if solution[v] > 0.5:
                vertices.append(1)

        edges = []
        for e in solver.edge_to_binary.values():
            if solution[e] > 0.5:
                edges.append(1)

        self.assertEqual(g1_solution.get_number_of_edges(), len(edges))
        self.assertEqual(g1_solution.get_number_of_vertices(), len(vertices))

        self.assertEqual(len(vertices), 4)
        self.assertEqual(len(edges), 2)
        self.assertEqual(len(dummys), 4)

        for e in g1_solution.get_edge_iterator():
            v0 = e.source()
            v1 = e.target()

            diff = np.abs(np.array(g1_solution.get_position(v0)) -\
                          np.array(g1_solution.get_position(v1)))

            self.assertTrue(np.all(diff == self.orientation))

if __name__ == "__main__":
    unittest.main()
