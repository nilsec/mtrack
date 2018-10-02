import os
import itertools
from collections import deque
import json
import signal

import multiprocessing
from mtrack.mt_utils import gen_config, read_config,  NoDaemonPool # Need outer non-daemon pool
from mtrack.preprocessing.create_probability_map import ilastik_get_prob_map
from mtrack import track

def generate_grid(param_dict,
                  grid_base_dir):

    """
    Generate directory structure and config files
    for the grid run and extract prob maps if needed. 
    Do Ilastik prob map predictions
    before to avoid redundant computations.
    """

    prob_map_params = {}
    if param_dict["extract_perp"][0]:
        pm_output_dir_perp = grid_base_dir + "/prob_maps" + "/perp"
        prob_map_params_perp = {"raw": param_dict["raw"][0],
                                "output_dir": pm_output_dir_perp,
                                "ilastik_source_dir": param_dict["ilastik_source_dir"][0],
                                "ilastik_project": param_dict["ilastik_project_perp"][0],
                                "file_extension": param_dict["file_extension"][0],
                                "h5_dset": param_dict["h5_dset"][0],
                                "label": param_dict["label"][0]}

        param_dict["extract_perp"] = [False]
        param_dict["pm_output_dir_perp"] = [pm_output_dir_perp]
        param_dict["perp_stack_h5"] = [ilastik_get_prob_map(**prob_map_params_perp)]
        
    else:
        assert(param_dict["perp_stack_h5"][0] is not None)
        

    if param_dict["extract_par"][0]:
        pm_output_dir_par = grid_base_dir + "/prob_maps" + "/par"
        prob_map_params_par = {"raw": param_dict["raw"][0],
                                "output_dir": pm_output_dir_par,
                                "ilastik_source_dir": param_dict["ilastik_source_dir"][0],
                                "ilastik_project": param_dict["ilastik_project_par"][0],
                                "file_extension": param_dict["file_extension"][0],
                                "h5_dset": param_dict["h5_dset"][0],
                                "label": param_dict["label"][0]}

        param_dict["extract_par"] = [False]
        param_dict["pm_output_dir_par"] = [pm_output_dir_par]
        param_dict["par_stack_h5"] = [ilastik_get_prob_map(**prob_map_params_par)]
    else:
        assert(param_dict["par_stack_h5"][0] is not None)

    grid = deque(dict(zip(param_dict, x))\
                 for x in itertools.product(*param_dict.itervalues()))

    n = 0
    while grid:
        params_n = grid.pop()
        
        # Change output dirs relative to base grid
        params_n["cfg_output_dir"] = grid_base_dir + "/grid_{}".format(n)
        
        params_n["eval_output_dir"] = params_n["eval_output_dir"].format(
                                                            params_n["cfg_output_dir"])        

        params_n["name_collection"] = params_n["name_collection"].format(n)
        
        params_n["chunk_output_dir"] = params_n["chunk_output_dir"].format(
                                                            params_n["cfg_output_dir"])

        params_n["cluster_output_dir"] = params_n["cluster_output_dir"].format(
                                                            params_n["cfg_output_dir"])
        
        params_n["validated_output_path"] = params_n["validated_output_path"].format(
                                                            params_n["cfg_output_dir"])
        try: 
            params_n["prob_map_chunks_perp_dir"] = params_n["prob_map_chunks_perp_dir"].format(
                                                            params_n["cfg_output_dir"])
 
            params_n["prob_map_chunks_par_dir"] = params_n["prob_map_chunks_par_dir"].format(
                                                            params_n["cfg_output_dir"])
        except AttributeError:
            pass
 
        gen_config(**params_n)
        n += 1
    

def run_grid(grid_base_dir, n_workers=8, skip_condition=lambda cfg_dict: False):
    grids = [os.path.join(grid_base_dir, f) for f in os.listdir(grid_base_dir) if f != "prob_maps"]
    grid_configs = [g + "/config.ini" for g in grids]

    print "Start grid search with {} workers on {} cpus...".format(n_workers, multiprocessing.cpu_count())

    if n_workers > 1:
        print "Working on MP branch..."
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        pool = NoDaemonPool(n_workers)
        signal.signal(signal.SIGINT, sigint_handler)

        try:
            results = []
            for cfg in grid_configs:
                cfg_dict = read_config(cfg)
                if not skip_condition(cfg_dict):
                    results.append(pool.apply_async(track, (cfg,)))
                else:
                    print "Skip {}...".format(os.path.dirname(cfg))
            for result in results:
                result.get(60*60*24*3)
        finally:
            pool.terminate()
            pool.join()

    else:
        print "Working on SP branch..."
        for cfg in grid_configs:
            cfg_dict = read_config(cfg)
            if not skip_condition(cfg_dict):
                track(cfg)
            else:
                print "Skip {}...".format(os.path.dirname(cfg))
