import os
import numpy as np
from grid import Grid
from postprocessing import skeletonize

if __name__ == "__main__":
    sol_base = "/media/nilsec/m1/gt_mt_data/solve_volumes/grid_2"
    solution_files = [os.path.join(sol_base, f + "/solution/volume.gt") for f in os.listdir(sol_base)]
    solution_files = [f for f in solution_files if os.path.exists(f)]

    grid_parameter = {"solution_file": solution_files,
                      "epsilon_lines": [75, 100, 125, 150, 175],
                      "epsilon_volumes": [75, 100, 125, 150],
                      "orientation_factor": [500, 1000, 1500],
                      "remove_singletons": [0, 3, 5]}

    output_dir = "/media/nilsec/m1/gt_mt_data/solve_skeletons/grid_1"

    solve_parameter = {"min_overlap_volumes": 1,
                       "canvas_shape": [30, 1025, 1025],
                       "offset": np.array([0,0,300]),
                       "output_dir": output_dir}

    grid = Grid(grid_parameter)
    grid_files = []
    try:
        grid_files = [os.path.join(output_dir,f) for f in os.listdir(output_dir) if f.endswith(".p")]
    except OSError:
        pass

    if grid_files:
        if os.path.getmtime(grid_files[0]) > os.path.getmtime(grid_files[1]):
            n = 0
        else:
            n = 1

        grid.load(grid_files[n])

    grid.run(f_solve=skeletonize,
             f_solve_parameter=solve_parameter,
             save_grid=True,
             verbose=True)
