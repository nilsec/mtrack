import unittest
import numpy as np
from mtrack.graphs import cost_converter, graph_converter, g1_graph, g2_solver
from mtrack import g2_to_g1_solution
from mtrack.mt_utils.graph_generator import GraphGenerator
from mtrack.solve import solve

class GraphSetUp(unittest.TestCase):
    def setUp(self):
        self.vertices = 100
        self.min_pos = 1
        self.max_pos = 11
        self.p_partner= 0.1

        self.generator = GraphGenerator(self.vertices,
                                        self.min_pos,
                                        self.max_pos,
                                        self.p_partner)

        self.g1 = self.generator.local_connectivity(5)

class DanglingVertexTestCase(GraphSetUp):
    def runTest(self):
        print "[Unittest]: Test dangling vertex selection/force..."
        for v in self.g1.get_vertex_iterator():
            assert(not self.g1.get_vertex_property("selected", v))
        for e in self.g1.get_edge_iterator():
            assert(not self.g1.get_edge_property("selected", e.source(), e.target()))

        self.selected_vertex = int(self.vertices/2.)        
        self.g1.select_vertex(self.selected_vertex)
        self.g1.solve_vertex(self.selected_vertex)

        gc = graph_converter.GraphConverter(self.g1)
        
        g2_vertices_N, g2_center_conflicts, g2_forced,\
        g2_vertex_forced, g2_edge_forced, index_maps = gc.get_mapping()

        self.assertFalse(g2_forced)
        self.assertFalse(g2_edge_forced)
        self.assertTrue(g2_vertex_forced)
        self.assertTrue(len(g2_vertex_forced) == 1)
        
        for g2v in g2_vertex_forced[0]:
            g1v_center = index_maps["g1_vertex_center"][g2v]
            self.assertTrue(g1v_center == self.selected_vertex)

        g2_graph, index_maps = gc.get_g2_graph()
        must_pick_one = g2_graph.get_must_pick_one() 
        self.assertTrue(len(must_pick_one) == 1)
        self.assertTrue(must_pick_one[0] == tuple(g2_vertex_forced[0]))
        for v in g2_graph.get_vertex_iterator():
            self.assertFalse(g2_graph.get_forced(v))

        g1_solved = solve(self.g1,
                          160.0,
                          0.0,
                          15.0,
                          16.0,
                          -80.0,
                          300,
                          voxel_size=[1.,1.,1.])

        v_in_solved = [v for v in g1_solved.get_vertex_iterator()]
        self.assertTrue(self.selected_vertex in v_in_solved)


class DanglingEdgeTestCase(GraphSetUp):
    def runTest(self):
        print "[Unittest]: Test dangling edge selection/force..."
        for v in self.g1.get_vertex_iterator():
            assert(not self.g1.get_vertex_property("selected", v))
        for e in self.g1.get_edge_iterator():
            assert(not self.g1.get_edge_property("selected", e.source(), e.target()))

        edges = [e for e in self.g1.get_edge_iterator()]
        self.selected_edge = edges[-1]

        self.g1.select_edge(self.selected_edge)
        self.g1.select_vertex(self.selected_edge.source())
        self.g1.select_vertex(self.selected_edge.target())
        self.g1.solve_edge(self.selected_edge)

        gc = graph_converter.GraphConverter(self.g1)
        
        g2_vertices_N, g2_center_conflicts, g2_forced,\
        g2_vertex_forced, g2_edge_forced, index_maps = gc.get_mapping()

        self.assertFalse(g2_forced)
        self.assertTrue(g2_edge_forced)
        self.assertFalse(g2_vertex_forced)
        self.assertTrue(len(g2_edge_forced) == 1)
        selected_edge = tuple(sorted([int(self.selected_edge.source()),\
                                     int(self.selected_edge.target())]))  
    
        for g2v in g2_edge_forced[0]:
            g1_edges = [tuple(sorted([int(e.source()), int(e.target())]))\
                        for e in index_maps["g2vertex_g1edges"][g2v]]

            self.assertTrue(selected_edge in g1_edges)

        g2_graph, index_maps = gc.get_g2_graph()
        must_pick_one = g2_graph.get_must_pick_one() 
        self.assertTrue(len(must_pick_one) == 1)
        self.assertTrue(must_pick_one[0] == tuple(g2_edge_forced[0]))
        for v in g2_graph.get_vertex_iterator():
            self.assertFalse(g2_graph.get_forced(v))

        g1_solved = solve(self.g1,
                          160.0,
                          0.0,
                          15.0,
                          16.0,
                          -80.0,
                          300,
                          voxel_size=[1.,1.,1.])

        e_in_solved = [tuple(sorted([int(e.source()), int(e.target())]))\
                       for e in g1_solved.get_edge_iterator()]

        self.assertTrue(selected_edge in e_in_solved)

class ForcedTestCase(GraphSetUp):
    def runTest(self):
        print "[Unittest]: Test forced..."
        for v in self.g1.get_vertex_iterator():
            assert(not self.g1.get_vertex_property("selected", v))
        for e in self.g1.get_edge_iterator():
            assert(not self.g1.get_edge_property("selected", e.source(), e.target()))

        done = False
        for v in self.g1.get_vertex_iterator():
            if done:
                continue
            incident_v = self.g1.get_incident_edges(v)
            if len(incident_v) > 1:
                self.selected_edge_pair = [incident_v[0], incident_v[1]]
                done = True

        for e in self.selected_edge_pair:
            self.g1.select_edge(e)
            self.g1.solve_edge(e)
            self.g1.select_vertex(e.source())
            self.g1.select_vertex(e.target())

        selected_edge_pair = [tuple(sorted([e.source(), e.target()]))\
                            for e in self.selected_edge_pair]

        gc = graph_converter.GraphConverter(self.g1)
        
        g2_vertices_N, g2_center_conflicts, g2_forced,\
        g2_vertex_forced, g2_edge_forced, index_maps = gc.get_mapping()

        self.assertTrue(g2_forced)
        self.assertFalse(g2_edge_forced)
        self.assertFalse(g2_vertex_forced)
        self.assertTrue(len(g2_forced) == 1)
   
        self.assertTrue(len(g2_forced) == 1)
        g2_forced = g2_forced[0]

        g1_edges = [tuple(sorted([int(e.source()), int(e.target())]))\
                    for e in index_maps["g2vertex_g1edges"][g2_forced]]

        for e in selected_edge_pair:
            self.assertTrue(e in g1_edges)

        for e in g1_edges:
            self.assertTrue(e in selected_edge_pair)

        g2_graph, index_maps = gc.get_g2_graph()
        must_pick_one = g2_graph.get_must_pick_one() 
        self.assertTrue(len(must_pick_one) == 0)

        for v in g2_graph.get_vertex_iterator():
            if v != g2_forced:
                self.assertFalse(g2_graph.get_forced(v))
            else:
                self.assertTrue(g2_graph.get_forced(v))

        g1_solved = solve(self.g1,
                          160.0,
                          0.0,
                          15.0,
                          16.0,
                          -80.0,
                          300,
                          voxel_size=[1.,1.,1.])

        e_in_solved = [tuple(sorted([int(e.source()), int(e.target())]))\
                       for e in g1_solved.get_edge_iterator()]

        for e in selected_edge_pair:
            self.assertTrue(e in e_in_solved)


if __name__ == "__main__":
    unittest.main()
