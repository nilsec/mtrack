import numpy as np
from opt_matching import OptMatch, get_lines
from process_solution import get_solution_lines, get_tracing_lines, get_lines
import os

def evaluate(tracing_file, 
             solution_file,
             chunk_size,
             distance_tolerance,
             dummy_cost,
             edge_selection_cost,
             pair_cost_factor,
             max_edges,
             voxel_size=[5.,5.,50.]):

    line_base_dir = None
    if cc_solution_dir[-1] == "/":
        line_base_dir = os.path.dirname(cc_solution_dir[:-1]) + "/lines"
    else:
        line_base_dir = os.path.dirname(cc_solution_dir) + "/lines"
        
    tracing_lines = get_lines(tracing_file, 
                                      line_base_dir + "/tracing/", 
                                      nml=True)

    rec_lines = get_lines(solution_file, 
                          line_base_dir + "/reconstruction/",
                          nml=True)

    """
    matcher = OptMatch(tracing_lines,
                       rec_lines,
                       chunk_size,
                       distance_tolerance,
                       dummy_cost,
                       edge_selection_cost,
                       pair_cost_factor,
                       max_edges)
    """
    
if __name__ == "__main__":
    tracing_file = "/media/nilsec/d0/gt_mt_data/test_tracing/v18_cropped_small_300_309.nml"
    cc_solution_dir = "/media/nilsec/d0/gt_mt_data/solve_volumes/test_volume_300_309/solution/volume.gt"

    evaluate(tracing_file, 
             cc_solution_dir,
             chunk_size=3,
             distance_tolerance=100.0,
             dummy_cost=100.0,
             edge_selection_cost=0.0,
             pair_cost_factor=1.0,
             max_edges=3)
