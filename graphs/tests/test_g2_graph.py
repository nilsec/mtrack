import sys
import os
sys.path.append(os.path.join('..', '..'))

import unittest
import numpy as np
from graphs import g2_graph

class SmallCircleGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.g2_vertices_N = 10
        self.g2 = g2_graph.G2(self.g2_vertices_N)

class AddGetConflictTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        self.g2.add_conflict(tuple([1, 3, 6]))
        self.g2.add_conflict(tuple([1, 4, 7]))
        self.g2.add_conflict(tuple([1, 4, 7]))

        self.assertEqual(set([tuple([1, 3, 6]), tuple([1, 4, 7])]), self.g2.get_conflicts())
        
        with self.assertRaises(AssertionError):
            self.g2.add_conflict(tuple([1, 0.5, 2]))
            self.g2.add_conflict(tuple([1000]))

class AddGetSumConstraints(SmallCircleGraphTestCase):
    def runTest(self):
        self.g2.add_sum_constraint([1, 3, 4], [1, 2, 5, 9])
        self.assertEqual([tuple([[1,3,4], [1,2,5,9]])], self.g2.get_sum_constraints())

        with self.assertRaises(AssertionError):
            self.g2.add_sum_constraint([1,3,4], [self.g2.get_number_of_vertices()])

class SetGetCost(SmallCircleGraphTestCase):
    def runTest(self):
        self.g2.set_cost(0, 100.2)
        self.assertEqual(self.g2.get_cost(0), 100.2)

        for j in range(1, self.g2.get_number_of_vertices()): 
            self.assertEqual(self.g2.get_cost(j), 0.0)
        

if __name__ == "__main__":
    unittest.main()


