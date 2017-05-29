import sys
import os
sys.path.append(os.path.join('..', '..'))

import unittest
import numpy as np
from g2_graph import g1_graph

class SmallCircleGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.g1_vertices_N = 10
        self.g1_edges_N = self.g1_vertices_N
        self.g1 = g1_graph.G1(self.g1_vertices_N)

        for j in range(0, self.g1_edges_N):
            self.g1.add_edge((j, (j+1) % self.g1_vertices_N))

class NumberOfEdgesTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        self.assertEqual(self.g1_edges_N, 
                         self.g1.get_number_of_edges(),
                         "Incorrect number of edges")

class NumberOfVerticesTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        self.assertEqual(self.g1_vertices_N,
                         self.g1.get_number_of_vertices(),
                         "Incorrect number of vertices")

class GetVertexTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        self.assertEqual(self.g1.get_vertex(3), 3)

class AddVertexTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        g1_vertices_before = self.g1.get_number_of_vertices()
        u = self.g1.add_vertex()
        g1_vertices_after = self.g1.get_number_of_vertices()

        self.assertEqual(g1_vertices_before,
                         g1_vertices_after - 1,
                         "Adding a vertex does not increase"+\
                         " vertex count by 1.")

        self.assertEqual(self.g1.get_number_of_edges(),
                         self.g1_edges_N,
                         "Number of edges changes when"+\
                         " adding a vertex.")

class GetEdgeTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        e_tuple_01 = self.g1.get_edge((0, 1))
        e_id_0 = self.g1.get_edge(0)

        self.assertIs(type(e_tuple_01), np.ndarray)
        self.assertIs(type(e_id_0), np.ndarray)
 
 

        e_tuple_12 = self.g1.get_edge((1, 2))
        e_id_1 = self.g1.get_edge(1)

        e_tuple_14 = self.g1.get_edge((1, 4))
        e_id_20 = self.g1.get_edge(20)

        self.assertTrue(np.all(e_id_0 == e_tuple_01))
        self.assertTrue(np.all(e_id_1 == e_tuple_12))

        self.assertEqual(e_tuple_14, None)
        self.assertEqual(e_id_20, None)

class AddEdgeTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        added_edge = self.g1.add_edge((1,4))
        self.assertIs(type(added_edge), np.ndarray)
 
        added_edge_id = (self.g1_edges_N - 1) + 1
        
        self.assertTrue(np.all(added_edge == self.g1.get_edge((1,4))))
        self.assertTrue(np.all(added_edge == self.g1.get_edge(added_edge_id)))

class GetEdgeMatrixTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        edge_matrix = self.g1.get_edge_matrix()
        #Test array type
        self.assertIs(type(edge_matrix), np.ndarray)
 
        edge_indices = [edge[2] for edge in edge_matrix]

        # This only holds if no adges are added or deleted after creation
        self.assertTrue(np.all(edge_indices == np.arange(self.g1_edges_N)))

        # Test what happens if an edge is added:
        self.g1.add_edge((1,4))
        edge_matrix_new = self.g1.get_edge_matrix()
        edge_indices_new = [edge[2] for edge in edge_matrix_new]
        # Fails: New ordering is [0,1,10,2,...]
        self.assertFalse(np.all(edge_indices_new == np.arange(self.g1_edges_N + 1)))


        # We can force the correct ordering [0, 1, 2, ..., 10] if we reinitialize 
        # after adding an edge:
        self.g1.g.reindex_edges()
        edge_matrix_reindex = self.g1.get_edge_matrix()
        edge_indices_reindex = [edge[2] for edge in edge_matrix_reindex] 
        # True:
        self.assertTrue(np.all(edge_indices_reindex == np.arange(self.g1_edges_N + 1)))

        # However, after reinitializing all id's change! Standard behavior after 
        # adding an edge is no reinit. Can be forced.

class GetNeighbourNodesTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        # Neighbours are ordered by vertex id.
        self.assertIs(type(self.g1.get_neighbour_nodes(1)), np.ndarray)
        self.assertTrue(np.all(self.g1.get_neighbour_nodes(1) == [0,2]))

class GetIncidentEdgesTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        incident_edges = self.g1.get_incident_edges(3)
        self.assertIs(type(incident_edges), np.ndarray)
        self.assertTrue(np.all(incident_edges[0] == np.array([3, 2, 2])))
        self.assertTrue(np.all(incident_edges[1] == np.array([3, 4, 3])))


if __name__ == "__main__":
    unittest.main()
