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
             rec_line_paths=None,
             time_limit=None):

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


    solution = matcher.solve(time_limit=time_limit)
    matcher.evaluate_solution(solution, tracing_line_paths, rec_line_paths, 
                                        os.path.dirname(line_base_dir) + "/evaluation")
