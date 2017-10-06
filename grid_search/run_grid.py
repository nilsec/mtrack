import solve
import os
from grid import Grid
from preprocessing import DirectionType


if __name__ == "__main__":

    ps_0 = DirectionType(0.35, 0.35)
    ps_1 = DirectionType(0.4, 0.4)
    ps_2 = DirectionType(0.45, 0.45)
    ps_3 = DirectionType(0.5, 0.5)

    grid_parameter = {"distance_threshold": [125, 150, 175],
                      "start_edge_prior": [140., 160., 180.],
                      "distance_factor": [0],
                      "orientation_factor":[15.0],
                      "comb_angle_factor":[16.0],
                      "selection_cost":[-70., -80., -90.],
                      "ps": [ps_0, ps_1, ps_2, ps_3]}

    prob_map_stack_file_perp_validation = "/media/nilsec/d0/gt_mt_data/" +\
                               "probability_maps/validation/perpendicular/stack/stack.h5"
    
    prob_map_stack_file_par_validation = "/media/nilsec/d0/gt_mt_data/" +\
                               "probability_maps/validation/parallel/stack/stack.h5"

    prob_map_stack = DirectionType(prob_map_stack_file_perp_validation,
                                   prob_map_stack_file_par_validation)

    output_dir = "/media/nilsec/d0/gt_mt_data/" +\
                 "solve_volumes/grid_2"
 
 
    solve_parameter = {"bounding_box": [300, 330],
                       "prob_map_stack": prob_map_stack,
                       "gs": DirectionType(0.5,0.5),
                       "voxel_size": [5.0,5.0,50.0],
                       "z_correction": 1,
                       "time_limit": 3000,
                       "output_dir": output_dir}

    grid = Grid(grid_parameter)
    
    grid_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".p")]

    if os.path.getmtime(grid_files[0]) > os.path.getmtime(grid_files[1]):
        n = 0
    else:
        n = 1

    grid.load(grid_files[n])
    grid.run(f_solve=solve.solve_bb_volume,
             f_solve_parameter=solve_parameter,
             save_grid=True,
             verbose=True,
             skip_runs=[35]) # Oom for run 10
