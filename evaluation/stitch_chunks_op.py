from opt_matching import OptMatch, interpolate_lines
from process_solution import get_lines
import os
import numpy as np

class Stitcher(object):
    def __init__(self):
        self.chunks = []

    def opt_match_chunks(self, 
                         chunk_1, 
                         chunk_2, 
                         output_dir,
                         chunk_size,
                         distance_tolerance,
                         dummy_cost,
                         pair_cost_factor,
                         max_edges,
                         edge_selection_cost,
                         time_limit):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        chunk_1_line_paths = get_lines(chunk_1, 
                                       os.path.join(output_dir, "chunk_1/lines/"),
                                       nml=True)

        chunk_2_line_paths = get_lines(chunk_2,
                                       os.path.join(output_dir, "chunk_2/lines/"),
                                       nml=True)

        chunk_1_itp, chunk_2_itp = interpolate_lines(chunk_1_line_paths,
                                                     chunk_2_line_paths)

        matcher = OptMatch(chunk_1_itp,
                           chunk_2_itp,
                           chunk_size,
                           distance_tolerance,
                           dummy_cost,
                           pair_cost_factor,
                           max_edges,
                           edge_selection_cost)

        matching = matcher.solve(time_limit=time_limit)

        matcher.evaluate_solution(matching, 
                                  chunk_1_line_paths,
                                  chunk_2_line_paths,
                                  os.path.join(output_dir, "matching"))

        
if __name__ == "__main__":
    stitcher = Stitcher()
    base_dir = "/media/nilsec/m1/gt_mt_data/solve_volumes/chunk_test_2"
    stitcher.opt_match_chunks(base_dir + "/solution2/volume.gt",
                              base_dir + "/solution3/volume.gt",
                              output_dir=base_dir + "/stitching23",
                              chunk_size=10,
                              distance_tolerance=55,
                              dummy_cost=10**8,
                              pair_cost_factor=10**8,
                              max_edges=1,
                              edge_selection_cost=-10.,
                              time_limit=1000)
