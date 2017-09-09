import unittest
import numpy as np
from graphs import cost_converter, graph_converter, g1_graph, g2_solver
from solve import solve, g2_to_g1_solution
from preprocessing import nml_io
from mt_l2 import solve_cc


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

class RealDataTestCase(unittest.TestCase):
    def setUp(self):
        self.g1 = g1_graph.G1(0)
        self.g1.load("./cc3117_min4_phy.gt")
        self.nml_g1 = "./cc3117_min4_phy.nml"

        nml_io.g1_to_nml(self.g1, self.nml_g1, voxel=False, voxel_size=[5.0, 5.0, 50.0])

        self.start_edge_prior = 180.0
        self.distance_factor = 0.0
        self.orientation_factor = 15.0
        self.comb_angle_factor = 16.0
        self.selection_cost = -100.0
        self.g_output_dir = "./3117_sol_g/"
        self.l_output_dir = "./3117_sol_l/"
        self.time_limit = 600
        self.voxel_size = [5.0, 5.0, 50.0]

class RandomGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.g_vertices_N = 3
        self.g_edges_N = np.random.randint(low=int(self.g_vertices_N/2.), 
                                           high=int(self.g_vertices_N * 2), 
                                           size=1)[0]
    
        self.g1 = g1_graph.G1(self.g_vertices_N)

        orientations = np.random.rand(self.g_vertices_N, 3)
        positions = np.random.rand(self.g_vertices_N, 3) * 10
        partner_mask = np.random.randint(low=0, high=2, size=self.g_vertices_N)
        partner = {v: -1 for v in self.g1.get_vertex_iterator()}
        
        for v in self.g1.get_vertex_iterator():
            if partner_mask[v]:
                if v < self.g_vertices_N - 1:
                    if partner[v] == -1:
                        partner[v] = int(v) + 1
                        partner[int(v) + 1] = v

        edges = set([])
        while len(edges) < self.g_edges_N:
            u, v = np.random.randint(0, self.g_vertices_N, 2)
                    

            if u > v:
                edges.add((u,v))

        for e in edges:
            self.g1.add_edge(e[0], e[1])

        v = 0
        for o in orientations:
            self.g1.set_orientation(v, o)
            v += 1

        v = 0
        for p in positions:
            self.g1.set_position(v, p)
            v += 1

        for v, p in partner.iteritems():
            self.g1.set_partner(int(v), int(p))

        self.nml_g1 = "./g1_random_test.nml"

        nml_io.g1_to_nml(self.g1, self.nml_g1, voxel=False, voxel_size=[5.0, 5.0, 50.0])

        self.start_edge_prior = 0.0
        self.distance_factor = 0.0
        self.orientation_factor = 0.0
        self.comb_angle_factor = 0.0
        self.selection_cost = -1.0
        self.g_output_dir = "./random_sol_g/"
        self.l_output_dir = "./random_sol_l/"
        self.time_limit = 600
        self.voxel_size = [5.0, 5.0, 50.0]

class DetGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.g_vertices_N = 3
        self.g_edges_N = 2
        
        self.g1 = g1_graph.G1(self.g_vertices_N)
        orientations = np.random.rand(self.g_vertices_N, 3)
        positions = np.random.rand(self.g_vertices_N, 3) * 10
        partner = {0:-1, 1:-1, 2:-1}
        edges = [(0,2), (1, 2)]

        for e in edges:
            self.g1.add_edge(e[0], e[1])

        v = 0
        for o in orientations:
            self.g1.set_orientation(v, o)
            v += 1

        v = 0
        for p in positions:
            self.g1.set_position(v, p)
            v += 1

        for v, p in partner.iteritems():
            self.g1.set_partner(int(v), int(p))

        self.nml_g1 = "./g1_det_test.nml"

        
        nml_io.g1_to_nml(self.g1, self.nml_g1, voxel=False, voxel_size=[5.0, 5.0, 50.0])

        self.start_edge_prior = 1.0
        self.distance_factor = 0.0
        self.orientation_factor = 1.0
        self.comb_angle_factor = 2.0
        self.selection_cost = -10.0
        self.g_output_dir = "./det_sol_g/"
        self.l_output_dir = "./det_sol_l/"
        self.time_limit = 600
        self.voxel_size = [5.0, 5.0, 50.0]
   
class TestCase(RealDataTestCase):
    def runTest(self):
        solve(self.g1,
              self.start_edge_prior,
              self.distance_factor,
              self.orientation_factor,
              self.comb_angle_factor,
              self.selection_cost,
              self.time_limit,
              self.g_output_dir,
              self.voxel_size)

        solve_cc(self.nml_g1,
                 self.start_edge_prior,
                 self.distance_factor,
                 self.orientation_factor,
                 self.comb_angle_factor,
                 self.selection_cost,
                 self.l_output_dir,
                 self.voxel_size,
                 True,
                 self.time_limit)
        

if __name__ == "__main__":
    unittest.main()
