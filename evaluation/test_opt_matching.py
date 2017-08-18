import unittest
import numpy as np
from opt_matching import OptMatch

class ParallelLines(unittest.TestCase):
    def setUp(self):
        self.line_gt = [np.array([10, 10, x]) for x in range(2,10)]
        self.line_rec = [np.array([12, 12, x]) for x in range(2, 10)]

class InitTestCase(ParallelLines):
    def runTest(self):
        matcher = OptMatch([self.line_gt], [self.line_rec], n=3, distance_tolerance=200.0)
        
        
if __name__ == "__main__":
    unittest.main()
