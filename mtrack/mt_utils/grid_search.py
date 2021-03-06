import os
import itertools
from collections import deque
import json
import signal
import time
import logging
import json

import multiprocessing
from mtrack.mt_utils import gen_config, read_config,  NoDaemonPool # Need outer non-daemon pool
from mtrack import track

logger = logging.getLogger(__name__)

def generate_grid(param_dict,
                  grid_base_dir,
                  n0=0):

    """
    Generate directory structure and config files
    for the grid run. 
    """

    grid = deque(dict(zip(param_dict, x))\
                 for x in itertools.product(*param_dict.itervalues()))

    n = n0
    while grid:
        params_n = grid.pop()
        
        # Change output dirs relative to base grid
        params_n["cfg_output_dir"] = grid_base_dir + "/grid_{}".format(n)
        
        params_n["name_collection"] = params_n["name_collection"].format(n)
        
        params_n["chunk_output_dir"] = params_n["chunk_output_dir"].format(
                                                            params_n["cfg_output_dir"])

        params_n["validated_output_path"] = params_n["validated_output_path"].format(
                                                            params_n["cfg_output_dir"])
 
        gen_config(**params_n)
        n += 1
    
def run_grid(grid_base_dir, n_workers=8, skip_condition=lambda cfg_dict: False):
    grids = [os.path.join(grid_base_dir, f) for f in os.listdir(grid_base_dir) if f != "prob_maps" and not f.endswith(".json")]
    grid_configs = [g + "/config.ini" for g in grids]
    grid_config_dicts = [read_config(cfg) for cfg in grid_configs]

    n_skipped = 0
    for cfg in grid_configs:
        cfg_dict = read_config(cfg)
        if skip_condition(cfg_dict):
            logger.info("Skip {}".format(os.path.dirname(cfg)))
            grid_configs.remove(cfg)
            n_skipped += 1
            
    logger.info("Skipped {} runs".format(n_skipped))
    logger.info("Start grid search with {} workers on {} cpus...".format(n_workers, multiprocessing.cpu_count()))
    logger.info("Grid size: {}".format(len(grids)))
    start = time.time()
    json.dump({"t0": start}, open(grid_base_dir + "/start.json", "w+"))

    if n_workers > 1:
        logger.info("Working on MP branch...")
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        pool = NoDaemonPool(n_workers)
        signal.signal(signal.SIGINT, sigint_handler)

        n_skipped = 0
        try:
            results = []
            for cfg in grid_config_dicts:
                results.append(pool.apply_async(track, (cfg,)))

            i = 1
            for result in results:
                result.get(60*60*24*14)
                elapsed = time.time() - start
                avg_time = elapsed/i
               
                logger.info("STATUS: Finished {}/{} in {} min. Average time per run: {} min".format(i, len(grid_configs), int(elapsed/60.), int(avg_time/60.)))
                logger.info("Total runs including skip runs done {}/{}".format(i + n_skipped, len(grids)))
                logger.info("Expected time till termination: {} min".format(int((avg_time * (len(grid_configs) - i))/60.)))
                i += 1

        finally:
            pool.terminate()
            pool.join()

    else:
        logger.info("Working on SP branch...")
        for cfg in grid_configs:
            cfg_dict = read_config(cfg)
            if not skip_condition(cfg_dict):
                track(cfg_dict)
            else:
                logger.info("Skip {}...".format(os.path.dirname(cfg)))
