import unittest
import numpy as np
from opt_matching import OptMatch

class ParallelLines(unittest.TestCase):
    def setUp(self):
        self.line_gt_1 = [np.array([10, 10, x]) for x in range(2,10)]
        self.line_gt_2 = [np.array([11, 11, x]) for x in range(0, 16)]
        self.line_rec = [np.array([12, 12, x]) for x in range(0, 20)]

class InitTestCase(ParallelLines):
    def runTest(self):
        matcher = OptMatch([self.line_gt_1, self.line_gt_2], [self.line_rec], n=3, 
                            distance_tolerance=1000.0, dummy_cost=1.0,\
                            edge_selection_cost=0.1, pair_cost_factor=5.0, max_edges=3)
        solution = matcher.solve()
        matcher.evaluate_solution(solution)
        
        
        
if __name__ == "__main__":
    unittest.main()
