import unittest
import numpy as np
from opt_matching import OptMatch, get_lines
import process_solution

class ParallelLines(unittest.TestCase):
    def setUp(self):
        self.line_gt_1 = [np.array([10, 10, x]) for x in range(2,10)]
        self.line_gt_2 = [np.array([11, 11, x]) for x in range(0, 16)]
        self.line_rec = [np.array([12, 12, x]) for x in range(0, 20)]

class InitTestCase(ParallelLines):
    def runTest(self):
        print "get lines..\n"
        lines_gt, lines_rec = get_lines("/media/nilsec/d0/gt_mt_data/experiments/selection_cost_grid0404_solve_4/lines", "/media/nilsec/d0/gt_mt_data/test_tracing/lines_v17_cropped")

        print "build ilp...\n"

        matcher = OptMatch(lines_gt, lines_rec, n=3, 
                            distance_tolerance=100.0, dummy_cost=3.0,\
                            edge_selection_cost=0.0, pair_cost_factor=5.0, max_edges=3)


        print "solve ilp..\n"
        solution = matcher.solve()
        matcher.evaluate_solution(solution)
        
        
        
if __name__ == "__main__":
    unittest.main()
