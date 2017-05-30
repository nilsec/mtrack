import sys
import os
sys.path.append(os.path.join('..', '..'))

import unittest
import numpy as np
from g2_graph import graph

class SmallCircleGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.g_vertices_N = 10
        self.g_edges_N = self.g_vertices_N
        self.g = graph.G(self.g_vertices_N)

        for j in range(0, self.g_edges_N):
            self.g.add_edge(j, (j+1) % self.g_vertices_N)

class NumberOfEdgesTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        self.assertEqual(self.g_edges_N, 
                         self.g.get_number_of_edges(),
                         "Incorrect number of edges")

class NumberOfVerticesTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        self.assertEqual(self.g_vertices_N,
                         self.g.get_number_of_vertices(),
                         "Incorrect number of vertices")

class GetVertexTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        self.assertEqual(self.g.get_vertex(3), 3)

        with self.assertRaises(ValueError):
            self.g.get_vertex(100)

class GetEdgeTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        e = self.g.get_edge(1, 2)
        self.assertEqual(e.source(), 1)
        self.assertEqual(e.target(), 2)

        edge_index_map = self.g.get_edge_index_map()
        self.assertEqual(edge_index_map[e.source(), e.target()], 1)


        with self.assertRaises(ValueError):
            self.g.get_edge(3, 5)


class AddVertexTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        g_vertices_before = self.g.get_number_of_vertices()
        u = self.g.add_vertex()
        g_vertices_after = self.g.get_number_of_vertices()

        self.assertEqual(g_vertices_before,
                         g_vertices_after - 1,
                         "Adding a vertex does not increase"+\
                         " vertex count by 1.")

        self.assertEqual(self.g.get_number_of_edges(),
                         self.g_edges_N,
                         "Number of edges changes when"+\
                         " adding a vertex.")

class AddEdgeTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        new_edge = self.g.add_edge(2, 6)
        edge_index_map = self.g.get_edge_index_map()

        self.assertEqual(edge_index_map[new_edge.source(), new_edge.target()], self.g_edges_N - 1 + 1)
        self.assertEqual(self.g.get_number_of_edges(), self.g_edges_N + 1)

        with self.assertRaises(AssertionError):
            self.g.add_edge(2, self.g.get_number_of_vertices())


class GetVertexIndexMapTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        vertex_index_map = self.g.get_vertex_index_map()
        
        for u in range(self.g_vertices_N):
            self.assertEqual(vertex_index_map[u], u)

        # Test what happens if we add a vertex:
        v = self.g.add_vertex()
        vertex_index_map = self.g.get_vertex_index_map()
        for u in range(self.g.get_number_of_vertices()):
            self.assertEqual(vertex_index_map[u], u)
        
        with self.assertRaises(ValueError):
            vertex_index_map[100]

class GetEdgeIndexMapTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        edge_index_map = self.g.get_edge_index_map()

        # Test that an exception is raised when a non existing edge
        # is requested
        with self.assertRaises(ValueError):
            edge_index_map[1, 4]

        e = self.g.get_edge(1, 2)
        self.assertEqual(edge_index_map[1, 2], edge_index_map[e])
        
class GetVertexIteratorTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        vertex_it = self.g.get_vertex_iterator()
        
        j = 0
        for v in vertex_it:
            self.assertEqual(v, j)
            j += 1

        self.g.add_vertex()
        vertex_it = self.g.get_vertex_iterator()
        j = 0
        for v in vertex_it:
            self.assertEqual(v, j)
            j += 1

class GetEdgeIteratorTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        edge_it = self.g.get_edge_iterator()
        
        for e in edge_it:
            self.assertEqual(e, self.g.get_edge(e.source(), e.target()))

class GetVertexArrayTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        vertex_array = self.g.get_vertex_array()
        self.assertTrue(np.all(vertex_array == np.arange(self.g_vertices_N)))
        
        self.g.add_vertex()
        vertex_array = self.g.get_vertex_array()
        self.assertTrue(np.all(vertex_array == np.arange(self.g_vertices_N + 1)))

class GetEdgeArrayTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        edge_array = self.g.get_edge_array()
        #Test array type
        self.assertIs(type(edge_array), np.ndarray)
 
        edge_indices = [edge[2] for edge in edge_array]

        # This only holds if no adges are added or deleted after creation
        self.assertTrue(np.all(edge_indices == np.arange(self.g_edges_N)))

        # Test what happens if an edge is added:
        self.g.add_edge(1,4)
        edge_array_new = self.g.get_edge_array()
        edge_indices_new = [edge[2] for edge in edge_array_new]
        # Fails: New ordering is [0,1,10,2,...]
        self.assertFalse(np.all(edge_indices_new == np.arange(self.g_edges_N + 1)))
        edge_index_map = self.g.get_edge_index_map()

        #NOTE: Index is the same in edge_index map.
        self.assertEqual(edge_index_map[1,4], 10)

        # We can force the correct ordering [0, 1, 2, ..., 10] if we reinitialize 
        # after adding an edge:
        self.g.g.reindex_edges()
        edge_array_reindex = self.g.get_edge_array()
        edge_indices_reindex = [edge[2] for edge in edge_array_reindex] 
        # True:
        self.assertTrue(np.all(edge_indices_reindex == np.arange(self.g_edges_N + 1)))

        # However, after reinitializing all id's change! Standard behavior after 
        # adding an edge is no reinit. Can be forced.

class NewVertexPropertyTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        vp = self.g.new_vertex_property("orientation", "vector<double>")
        self.assertIsNotNone(vp)
        self.assertTrue(vp[0] == np.array([]))

class SetGetVertexPropertyTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        vp = self.g.new_vertex_property("orientation", "vector<double>")
        orientation = np.array([1.0, 0.8888, 0.11101])
        self.g.set_vertex_property("orientation", 1, orientation)
        self.assertTrue(np.all(self.g.get_vertex_property("orientation", 1) == orientation))

        with self.assertRaises(KeyError):
            self.g.get_vertex_property("diesdas", 1)

        with self.assertRaises(ValueError):
            self.g.get_vertex_property("orientation", 100)

class NewEdgePropertyTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        ep = self.g.new_edge_property("distance", "double")
        self.assertIsNotNone(ep)
        for e in self.g.get_edge_iterator():
            self.assertTrue(ep[e] == 0.0)

class SetGetEdgePropertyTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        ep = self.g.new_edge_property("distance", "double")
        distance = 12.5555
        self.g.set_edge_property("distance", 1, 2, distance)
        self.assertTrue(self.g.get_edge_property("distance", 1, 2) == distance)

        with self.assertRaises(KeyError):
            self.g.get_edge_property("diesdas", 1, 2)

        with self.assertRaises(ValueError):
            self.g.get_edge_property("distance", 1, 1000)

class GraphPropertyTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        gp = self.g.new_graph_property("test", "python::object")
        pyobj = {1: 2, 2:1, 3: -1, 4 :-1} 
        self.g.set_graph_property("test", pyobj)
        self.assertEqual(self.g.get_graph_property("test"), pyobj)
        
class GetNeighbourNodesTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        # Neighbours are ordered by vertex id.
        self.assertIs(type(self.g.get_neighbour_nodes(1)), np.ndarray)
        self.assertTrue(np.all(self.g.get_neighbour_nodes(1) == [0,2]))

class GetIncidentEdgesTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        incident_edges = self.g.get_incident_edges(3)
        self.assertIs(type(incident_edges), np.ndarray)
        self.assertTrue(np.all(incident_edges[0] == np.array([3, 2, 2])))
        self.assertTrue(np.all(incident_edges[1] == np.array([3, 4, 3])))

if __name__ == "__main__":
    unittest.main()
