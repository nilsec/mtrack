import unittest
import numpy as np
from opt_matching import OptMatch
import process_solution

class ParallelLines(unittest.TestCase):
    def setUp(self):
        self.line_gt_1 = [np.array([10, 10, x]) for x in range(2,10)]
        self.line_gt_2 = [np.array([11, 11, x]) for x in range(0, 16)]
        self.line_rec = [np.array([12, 12, x]) for x in range(2, 20)]

class InitTestCase(ParallelLines):
    def runTest(self):
        matcher = OptMatch([self.line_gt_1], [self.line_rec], n=3, 
                            distance_tolerance=50.0, dummy_cost=100.0,\
                            edge_selection_cost=-10.0, pair_cost_factor=5.0, max_edges=3)

        
        for gt_node in self.line_gt_1:
            for rec_node in self.line_rec:
                print gt_node[2], rec_node[2], np.linalg.norm((rec_node-gt_node) * [5.,5.,50.]) 


        print "solve ilp..\n"
        solution = matcher.solve()
        matcher.evaluate_solution(solution)
        
        
        
if __name__ == "__main__":
    unittest.main()
