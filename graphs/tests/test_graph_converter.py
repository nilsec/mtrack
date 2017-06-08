import sys
import os
sys.path.append(os.path.join('..', '..'))

import unittest
import numpy as np
from graphs import graph_converter
from graphs import g1_graph
from graphs import g2_graph
from mt_l2 import L1Graph, L2Graph, l1_to_l2

class SmallG1CircleTestCase(unittest.TestCase):
    def setUp(self):
        self.g1_vertices_N = 20
        self.g1 = g1_graph.G1(self.g1_vertices_N)
        
        for j in range(0, self.g1_vertices_N):
            self.g1.add_edge(j, (j+1) % self.g1_vertices_N)

        orientation_0 = np.array([1.0, 2.0, 3.0])
        position_0 = np.array([1.0, 10.0, 100.0])
        partner = {0: 2, 2: 0}
        self.resulting_conflict_edge_ids = [(0, 1), (0, 2), (1, 19), (2, 19)]
        
        for j in range(0, self.g1_vertices_N):
            self.g1.set_orientation(j, orientation_0 * (j + 1))
            self.g1.set_position(j, position_0 * (j + 1))
            try:
                self.g1.set_partner(j, partner[j])
            except KeyError:
                self.g1.set_partner(j, -1)

        """
        In the circle case without partner:
        1. N g_2 nodes.
        2. N * 2 start end nodes
        -> 3N g2_nodes

        1. conflict -> 3N - 1 g2 nodes
        """
        self.expected_g2_vertices_N = 59

        """
        In the circle case w.o. partner:
        3 conflicts per node:
        N conflict sets and total 3*N conflicts.
        
        With partner we can not form 1 g2 node:
        N conflict sets and total 3*N-1 conflicts

        """

        self.expected_conflict_sets = self.g1_vertices_N
        self.total_nodes_in_conflict = self.g1_vertices_N * 3 - 1
 

class SmallChainTestCase(unittest.TestCase):
    def setUp(self):
        self.g1_vertices_N = 5
        self.g1 = g1_graph.G1(self.g1_vertices_N)

        self.l1 = L1Graph(self.g1_vertices_N)
        
        for j in range(0, self.g1_vertices_N - 1):
            self.g1.add_edge(j, j+1)
            self.l1.add_edge(j, j+1)


        orientation_0 = np.array([1.0, 2.0, 3.0])
        position_0 = np.array([1.0, 10.0, 100.0])
        partner = {0: 2, 2: 0}
        #partner = {}
        self.resulting_conflict_edge_ids = [(0, 1), (0, 2)]
        
        for j in range(0, self.g1_vertices_N):
            self.g1.set_orientation(j, orientation_0 * (j + 1))
            self.g1.set_position(j, position_0 * (j + 1))
            try:
                self.g1.set_partner(j, partner[j])
            except KeyError:
                self.g1.set_partner(j, -1)

            self.l1.orientations[j] = orientation_0 * (j + 1)
            self.l1.positions[j] = position_0 * (j  + 1)
            try:
                self.l1.partner[j] = partner[j]
            except KeyError:
                self.l1.partner[j] = -1  

        """
        FOR NO PARTNER CONFLICTS:

        For a g1 chain with start and end point:
        0---1---2---3---4---5
        we expect the following number of
        g2 nodes:
        1. Each center vertex has one g2 node:
              N - 2

        2. Each center vertex has TWO g2 start/end nodes:
              2 * (N -2)

        3. The border nodes have ONE g2 start/end node:
              2

        -> N -2 + 2 + 2 * (N - 2) = 3N - 4
        
        For N=5 this equals 11 g2 nodes

        If we include a partner in the g1 graph
        like done above, 0:2, 2:0
        we can not form the g2 node:
        0---1---2 Thus: 
        
        for N=5 we expect 10 g2 nodes with a 
        partner constraint at the chain boarder.
        """
        self.expected_g2_vertices_N = 10

        """
        How many center conflicts do we expect?
        
        Without partner and chain topology we have:
        1. For the non end nodes we always get
           3 g2 nodes that share a particular 
           g1 node as the center node.
           Those are the trivial one e.g. 1--2--3
           plus the two end g2 nodes 1--2--s
           and s--2--3:

           N-2 conflict sets

        2. The edge nodes can not conflict with anything:
           0 conflict sets.

        Thus we expect N-2 conflict sets containing
        in total N-2 * 3 = 3N - 6 nodes

        Influence of partner nodes:
        We can not form one g2 node.
        In this setup 0--1--2 is not possible
        but 0--1--s and s--1--2 is still possible
        and in conflict
        thus we expect N - 3 conflict sets where
        one conflict set contains 2 node ids
        and 2 conflict sets with 3 node ids

        -> 3 conflict sets with 8 conflicts total
        """
        self.expected_conflict_sets = 3
        self.total_nodes_in_conflict = 8

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
    
 
class CheckEdgePartnerConflictTestCase(SmallG1CircleTestCase):
    def runTest(self):
        converter = graph_converter.GraphConverter(self.g1)
        edge_index_map = self.g1.get_edge_index_map()

        for e1 in self.g1.get_edge_iterator():
            for e2 in self.g1.get_edge_iterator():
                e1_id = edge_index_map[e1]
                e2_id = edge_index_map[e2]

                if e1_id > e2_id:
                    conflict = converter.check_edge_partner_conflict(e1, e2)
                    if (e2_id, e1_id) in self.resulting_conflict_edge_ids:                        
                        #print "Conflict", e2_id, e1_id
                        self.assertTrue(conflict)
 
                    else:
                        #print "No Conf", e2_id, e1_id, "\n"
                        self.assertFalse(conflict)

class GetMappingTestCase(SmallG1CircleTestCase):
    def runTest(self):
        converter = graph_converter.GraphConverter(self.g1)
        g2_vertices_N, g2_center_conflicts, index_maps = converter.get_mapping()
        self.assertEqual(g2_vertices_N, self.expected_g2_vertices_N)

        self.assertEqual(len(g2_center_conflicts), self.expected_conflict_sets)
        self.assertEqual(sum([len(c_set) for c_set in g2_center_conflicts]), 
                         self.total_nodes_in_conflict)

        for key, value in index_maps["g1_vertex_center"].iteritems():
            self.assertTrue(key in index_maps["g1_vertex_center_inverse"][value])

        """
        print "Center Conflicts:\n"
        for g in g2_center_conflicts:
            print g

        print "\n\ng2vertex_g1edges:\n"
        for g2_v, g1_e in index_maps["g2vertex_g1edges"].iteritems():
            print g2_v, ": ", "e1: ({}, {}), e2: ({}, {})".format(g1_e[0].source(),
                                                                  g1_e[0].target(),
                                                                  g1_e[1].source(),
                                                                  g1_e[1].target())
        """
class GetPartnerConflictsTestCase(SmallChainTestCase):
    def runTest(self):
        """
        Extend
        """
        converter = graph_converter.GraphConverter(self.g1)
        g2_vertices_N, g2_center_conflicts, index_maps = converter.get_mapping()
        g2 = g2_graph.G2(g2_vertices_N)

        g2_partner_conflicts = converter.get_partner_conflicts(g2, 
                               index_maps["g1_vertex_center_inverse"])

        """
        print g2_partner_conflicts

        for g2_v, g1_e in index_maps["g2vertex_g1edges"].iteritems():
            print g2_v, ": ", "e1: ({}, {}), e2: ({}, {})".format(g1_e[0].source(),
                                                                  g1_e[0].target(),
                                                                  g1_e[1].source(),
                                                                  g1_e[1].target())
        """

class GetContinuationConstraintsTestCase(SmallChainTestCase):
    def runTest(self):
        """
        Extend
        """
        converter = graph_converter.GraphConverter(self.g1)
        g2_vertices_N, g2_center_conflicts, index_maps = converter.get_mapping()
  
        continuation_constraints = converter.get_continuation_constraints(
                                   index_maps["g1edge_g2vertices"],
                                   index_maps["g1_vertex_center"])


        """
        for c in continuation_constraints:
            print c

        for g2_v, g1_e in index_maps["g2vertex_g1edges"].iteritems():
            print g2_v, ": ", "e1: ({}, {}), e2: ({}, {})".format(g1_e[0].source(),
                                                                  g1_e[0].target(),
                                                                  g1_e[1].source(),
                                                                  g1_e[1].target())
        """

class GetG2GraphTestCase(SmallChainTestCase):
    def runTest(self):
        converter = graph_converter.GraphConverter(self.g1)
        g2, index_maps = converter.get_g2_graph()

        l2, l2vertex_l1edges, l1_vertex_center = l1_to_l2(self.l1)
        g2_vertex_index_map = g2.get_vertex_index_map()
        g1_vertex_index_map = self.g1.get_vertex_index_map()
        

        g2_test_dict = {}
        for g2_v, g1_e in index_maps["g2vertex_g1edges"].iteritems():
            print g2_v, g1_e
            g1_e_0_s = self.g1.get_vertex_id(g1_e[0].source(), 
                                             g1_vertex_index_map)
            g1_e_0_t =  self.g1.get_vertex_id(g1_e[0].target(), 
                                              g1_vertex_index_map)
 
            g1_e_1_s = self.g1.get_vertex_id(g1_e[1].source(), 
                                             g1_vertex_index_map)
 
            g1_e_1_t = self.g1.get_vertex_id(g1_e[1].target(), 
                                             g1_vertex_index_map)
 
            g2_test_dict[g2_vertex_index_map[g2_v]] = [sorted([g1_e_0_s, g1_e_0_t]), 
                                                       sorted([g1_e_1_s, g1_e_1_t])]

 
        l2_test_dict = {}
        for g2_v, g1_e in  l2vertex_l1edges.iteritems():
            if g1_e[0] != (-1):
                e0 = list(self.l1.edges[g1_e[0]])
            else:
                e0 = [-1, -1]

            if g1_e[1] != (-1):
                e1 = list(self.l1.edges[g1_e[1]])
            else:
                e1 = [-1, -1]

            l2_test_dict[g2_v] = [sorted(e0), sorted(e1)]
            
        self.assertEqual(g2_test_dict, l2_test_dict)
    
if __name__ == "__main__":
    unittest.main()
