import numpy as np
import os

import pylp

from mtrack.graphs import G1
from mtrack.evaluation.opt_matching import OptMatch, interpolate_lines
from mtrack.evaluation.process_solution import get_lines

def evaluate(tracing_file, 
             solution_file,
             scale_solution_to_voxel,
             chunk_size,
             distance_tolerance,
             dummy_cost,
             edge_selection_cost,
             pair_cost_factor,
             max_edges,
             voxel_size,
             output_dir,
             tracing_line_paths=None,
             rec_line_paths=None,
             time_limit=None):

    
    if tracing_line_paths is None:    
        tracing_line_paths = get_lines(volume=tracing_file, 
                                       output_dir=output_dir + "/lines/tracing/", 
                                       scale=False,
                                       voxel_size=voxel_size,
                                       nml=True)

    if rec_line_paths is None:
        rec_line_paths = get_lines(volume=solution_file, 
                                   output_dir=output_dir + "/lines/reconstruction/",
                                   scale=scale_solution_to_voxel,
                                   voxel_size=voxel_size,
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
    
    solution = matcher.solve(time_limit=time_limit)
    matcher.evaluate_solution(solution, 
                              tracing_line_paths, 
                              rec_line_paths, 
                              output_dir + "/evaluation")
