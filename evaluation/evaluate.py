import numpy as np
from opt_matching import OptMatch, interpolate_lines
from process_solution import get_lines
from graphs import G1
import os
import pdb
import pylp

def evaluate(tracing_file, 
             solution_file,
             chunk_size,
             distance_tolerance,
             dummy_cost,
             edge_selection_cost,
             pair_cost_factor,
             max_edges,
             voxel_size=[5.,5.,50.],
             tracing_line_paths=None,
             rec_line_paths=None):

    line_base_dir = None
    if solution_file[-1] == "/":
        line_base_dir = os.path.dirname(solution_file[:-1]) + "/lines"
    else:
        line_base_dir = os.path.dirname(solution_file) + "/lines"

    
    if tracing_line_paths is None:    
        tracing_line_paths = get_lines(tracing_file, 
                              line_base_dir + "/tracing/", 
                              nml=True)

    if rec_line_paths is None:
        rec_line_paths = get_lines(solution_file, 
                          line_base_dir + "/reconstruction/",
                          nml=True)

    rec_lines, tracing_lines = interpolate_lines(rec_line_paths, tracing_line_paths)

    matcher = OptMatch(tracing_lines,
                       rec_lines,
                       chunk_size,
                       distance_tolerance,
                       dummy_cost,
                       pair_cost_factor,
                       max_edges,
                       edge_selection_cost)
    
    print "LOGLEVEL:", pylp.getLogLevel()


    solution = matcher.solve()
    matcher.evaluate_solution(solution, tracing_line_paths, rec_line_paths, 
                                        os.path.dirname(line_base_dir) + "/evaluation")
    
if __name__ == "__main__":
    tracing_file = "/media/nilsec/d0/gt_mt_data/test_tracing/v18_cropped_small_300_309.nml"
    solution_file_0404 = "/media/nilsec/d0/gt_mt_data/solve_volumes/test_volume_300_309/solution/volume.gt"
    solution_file_0304 = "/media/nilsec/d0/gt_mt_data/solve_volumes/test_volume_ps0304_300_309/solution/volume.gt"

    solution_file_0303 = "/media/nilsec/d0/gt_mt_data/solve_volumes/test_volume_ps0303_300_309/solution/volume.gt"
 

    tracing_line_paths = ["/media/nilsec/d0/gt_mt_data/solve_volumes/test_volume_300_309/solution/minimal_lines/tracing/cc33_min1_phy.gt"]

    rec_line_paths = ["/media/nilsec/d0/gt_mt_data/solve_volumes/test_volume_300_309/solution/minimal_lines/reconstruction/cc6_min1_phy.gt"]

    evaluate(tracing_file, 
             solution_file_0303,
             chunk_size=10,
             distance_tolerance=100.0,
             dummy_cost=100000,
             edge_selection_cost=-10.0,
             pair_cost_factor=1.0,
             max_edges=5,
             tracing_line_paths=None,
             rec_line_paths=None)
