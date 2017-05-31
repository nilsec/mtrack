import sys
import os
sys.path.append(os.path.join('..', '..'))

import unittest
import numpy as np
from g2_graph import g1_graph

class SmallCircleGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.g1_vertices_N = 10
        self.g1 = g1_graph.G1(self.g1_vertices_N)

class SetGetOrientationTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        orientation = np.array([1.0, 0.11111, 0.123])
        
        for n in range(self.g1.get_number_of_vertices()):
            self.g1.set_orientation(n, orientation * n)

        for n in range(self.g1.get_number_of_vertices()):
            self.assertTrue(np.all(orientation * n == self.g1.get_orientation(n)))

class SetGetPositionTestCase(SmallCircleGraphTestCase):
    def runTest(self):
        position = np.array([2.0, 340.0, 123])
        
        for n in range(self.g1.get_number_of_vertices()):
            self.g1.set_position(n, position * n)

        for n in range(self.g1.get_number_of_vertices()):
            self.assertTrue(np.all(position * n == self.g1.get_position(n)))

class SetGetPartnerTestCase(SmallCircleGraphTestCase):
    def runTest(self):
         
        for n in range(self.g1.get_number_of_vertices()):
            self.g1.set_partner(n, (n+1) % self.g1.get_number_of_vertices())

        for n in range(self.g1.get_number_of_vertices()):
            self.assertTrue(np.all((n+1) % self.g1.get_number_of_vertices() == self.g1.get_partner(n)))

if __name__ == "__main__":
    unittest.main()
