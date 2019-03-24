from mtrack.cores import CoreSolver, CoreBuilder, DB
from mtrack.preprocessing import g1_to_nml, Chunker, extract_maxima_candidates
from mtrack.mt_utils import read_config, check_overlap
from mtrack.graphs.g1_graph import G1
from solve import solve

import numpy as np
import h5py
import os
import multiprocessing, logging
import sys
import signal
import traceback
import functools
import shutil
import pdb


def track(config):
    roi = [config["roi_x"], config["roi_y"], config["roi_z"]]
    volume_offset = config["volume_offset"] * config["voxel_size"]

    roi_volume_size = np.array([r[1] - r[0] for r in roi]) * config["voxel_size"]
    roi_offset = np.array([r[0] for r in roi]) * config["voxel_size"]

    x_lim_roi = {"min": roi_offset[0],
                 "max": roi_offset[0] + roi_volume_size[0]}

    y_lim_roi = {"min": roi_offset[1],
                 "max": roi_offset[1] + roi_volume_size[1]}

    z_lim_roi = {"min": roi_offset[2],
                 "max": roi_offset[2] + roi_volume_size[2]}

    db_credentials = config["db_credentials"]


    if np.any(config["context_size"] * config["voxel_size"] < 2 * config["distance_threshold"]):
        raise ValueError("The context size needs to be at least " +\
                         "twice as large as the distance threshold in all dimensions")

    # Init logger:
    logging.info("Start tracking")

    # Generate core geometry: 
    builder = CoreBuilder(volume_size=roi_volume_size,
                          core_size=config["core_size"] * config["voxel_size"],
                          context_size=config["context_size"] * config["voxel_size"],
                          offset=roi_offset)
    
    cores = builder.generate_cores()
    
    # Get conflict free core lists
    cf_lists = builder.gen_cfs()

    if config["extract_candidates"]:
        max_chunk_dir = chunk_prob_map(volume_shape=config["volume_shape"],
                                    max_chunk_shape=config["max_chunk_shape"],
                                    volume_offset=config["volume_offset"],
                                    voxel_size=config["voxel_size"],
                                    prob_map_h5=config["maxima"],
                                    dset=config["maxima_dset"],
                                    output_dir=os.path.join(config["chunk_output_dir"], "maxima"))

        pm_chunk_dir = chunk_prob_map(volume_shape=config["volume_shape"],
                                    max_chunk_shape=config["max_chunk_shape"],
                                    volume_offset=config["volume_offset"],
                                    voxel_size=config["voxel_size"],
                                    prob_map_h5=config["prob_map"],
                                    dset=config["prob_map_dset"],
                                    output_dir=os.path.join(config["chunk_output_dir"], "pm"))

        max_chunks = [os.path.join(max_chunk_dir, f)\
                      for f in os.listdir(max_chunk_dir) if f.endswith(".h5")]

        pm_chunks = [os.path.join(pm_chunk_dir, f)\
                     for f in os.listdir(pm_chunk_dir) if f.endswith(".h5")]

        config["pm_chunks"] = pm_chunks
        config["max_chunks"] = max_chunks

        """
        Extract id, and volume information from chunks
        and compare with ROI
        """
        chunk_limits = {}
        chunk_ids = {}
        roi_pm_chunks = []
        roi_max_chunks = []
        for max_chunk, pm_chunk in zip(max_chunks, pm_chunks):
            if not os.path.isfile(pm_chunk):
                raise ValueError("{} is not a file".format(pm_chunk))

            f = h5py.File(pm_chunk, "r")
            attrs = f[config["prob_map_dset"]].attrs.items()
            f.close()

            chunk_limit = attrs[1][1]
            chunk_id = attrs[0][1]

            chunk_limits[pm_chunk] = chunk_limit 
            chunk_ids[pm_chunk] = chunk_id

            full_ovlp = np.array([False, False, False])
            for i in range(3):
                full_ovlp[i] = check_overlap(chunk_limit[i], roi[i])

            if np.all(full_ovlp):
                roi_pm_chunks.append(pm_chunk)
                roi_max_chunks.append(max_chunk)

        """
        Extract candidates from all ROI chunks and write to specified
        database.
        """

        write_candidate_graph(max_chunks=max_chunks,
                              max_dset=config["maxima_dset"],
                              pm_chunks=pm_chunks,
                              pm_dset=config["prob_map_dset"],
                              name_db=config["name_db"],
                              collection=config["name_collection"],
                              db_credentials=config["db_credentials"],
                              distance_threshold=config["distance_threshold"],
                              voxel_size=config["voxel_size"],
                              cores=cores,
                              cf_lists=cf_lists,
                              volume_offset=config["volume_offset"],
                              overwrite=True,
                              mp=config["mp"])

        
        # Clean up chunks
        shutil.rmtree(max_chunk_dir)
        shutil.rmtree(pm_chunk_dir)

        """
        Solve the ROI and write to specified database. The result
        is written out depending on the options in the Output section
        of the config file.
        """
    if config["reset"]:
        db = DB(config["db_credentials"])
        db.reset_collection(config["name_db"], 
                            config["name_collection"])

    if config["solve"]:
        solve_candidate_volume(name_db=config["name_db"],
                               collection=config["name_collection"],
                               db_credentials=config["db_credentials"],
                               cc_min_vertices=config["cc_min_vertices"],
                               start_edge_prior=config["start_edge_prior"],
                               selection_cost=config["selection_cost"],
                               orientation_factor=config["orientation_factor"],
                               comb_angle_factor=config["comb_angle_factor"],
                               time_limit=config["time_limit_per_cc"],
                               cores=cores,
                               cf_lists=cf_lists,
                               voxel_size=config["voxel_size"],
                               offset=np.array(roi_offset),
                               mp=config["mp"],
                               backend=config["backend"])

    
    if config["validate_selection"]:
        db = DB(config["db_credentials"])
        try:
            g1_selected = db.validate_selection(name_db=config["name_db"],
                                                collection=config["name_collection"],
                                                x_lim=x_lim_roi,
                                                y_lim=y_lim_roi,
                                                z_lim=z_lim_roi)
        except ValueError:
            logging.warning("WARNING, solution contains no vertices!")
            g1_selected = G1(0)

        if config["export_validated"]:
            g1_to_nml(g1_selected, 
                      config["validated_output_path"],
                      knossos=True,
                      voxel_size=config["voxel_size"])

def chunk_prob_map(volume_shape,
                   max_chunk_shape,
                   volume_offset,
                   voxel_size,
                   prob_map_h5,
                   dset,
                   output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunker = Chunker(volume_shape,
                      max_chunk_shape,
                      voxel_size,
                      offset=volume_offset)

    chunks = chunker.chunk()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(prob_map_h5, "r") as f0:
        data = f0[dset]
        assert(len(data.shape) == 3)

        for chunk in chunks:
            limits = chunk.limits
            chunk_data = np.array(data[limits[2][0] - volume_offset[2]:limits[2][1] - volume_offset[2],
                                       limits[1][0] - volume_offset[1]:limits[1][1] - volume_offset[1],
                                       limits[0][0] - volume_offset[0]:limits[0][1] - volume_offset[0]])

            f = h5py.File(output_dir + "/chunk_{}.h5".format(chunk.id), 'w')
            f.create_dataset(dset, data=chunk_data)
            f[dset].attrs.create("chunk_id", chunk.id)
            f[dset].attrs.create("limits", limits)
            f.close()

    return output_dir

def connect_candidates_alias(db, 
                             name_db,
                             collection,
                             x_lim,
                             y_lim,
                             z_lim,
                             distance_threshold):
    """
    Alias for instance methods that allows us
    to call them in a pool
    """
    
    db.connect_candidates(name_db,
                          collection,
                          x_lim,
                          y_lim,
                          z_lim,
                          distance_threshold)

def write_candidate_graph(max_chunks,
                          max_dset,
                          pm_chunks,
                          pm_dset,
                          name_db,
                          collection,
                          db_credentials,
                          distance_threshold,
                          voxel_size,
                          cores,
                          cf_lists,
                          volume_offset,
                          overwrite=False,
                          mp=True):

    logging.info("Extract candidates...")
    db = DB(db_credentials)
    n_chunk = 0
    id_offset = 1
    
    # Overwrite if necesseray:
    graph = db.get_collection(name_db, collection, overwrite=overwrite)
    for chunk in max_chunks:
        logging.info("Extract chunk {}/{}...".format(n_chunk, len(max_chunks)))

        f = h5py.File(chunk, "r")
        attrs = f[max_dset].attrs.items()
        f.close()

        chunk_limits = attrs[1][1]
        offset_chunk = [chunk_limits[0][0], 
                        chunk_limits[1][0], 
                        chunk_limits[2][0]]

        candidates = extract_maxima_candidates(chunk,
                                               max_dset,
                                               offset_chunk,
                                               id_offset)

        id_offset_tmp = db.write_candidates(name_db,
                                            collection,
                                            candidates,
                                            voxel_size,
                                            overwrite=False)
    
        assert(graph.find({"selected": {"$exists": True}}).count() == id_offset_tmp)

        id_offset = id_offset_tmp + 1
        n_chunk += 1

    # Don't forward SIGINT to child processes
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool()
    signal.signal(signal.SIGINT, sigint_handler)

    bound_connect_candidates_alias = functools.partial(connect_candidates_alias, db)

    logging.info("Connect candidates...")
    try:
        for cf_core_ids in cf_lists:
            logging.info("Connecting {}".format(cf_core_ids))
                        
            results = []
            if mp:
                for core_id in cf_core_ids:
                    logging.info("Add context {} to pool (mp: {})".format(core_id, mp))
                    core = cores[core_id]
                    results.append(pool.apply_async(bound_connect_candidates_alias, 
                                                    (name_db,
                                                     collection,
                                                     core.x_lim_context,
                                                     core.y_lim_context,
                                                     core.z_lim_context,
                                                     distance_threshold,
                                                     ))
                                  )

                # Catch exceptions and SIGINTs
                for result in results:
                    result.get(60*60*24*3) 
            else:
                for core_id in cf_core_ids:
                    core = cores[core_id]
                    results.append(db.connect_candidates(name_db,
                                                         collection,
                                                         core.x_lim_context,
                                                         core.y_lim_context,
                                                         core.z_lim_context,
                                                         distance_threshold,
                                                         ))
    finally:
        pool.terminate()
        pool.join()

    logging.info("Add edge costs...")
    for chunk in pm_chunks:
        logging.info("Work on chunk {}/{}...".format(n_chunk, len(pm_chunks)))

        f = h5py.File(chunk, "r")
        attrs = f[pm_dset].attrs.items()
        shape = np.shape(np.array(f[pm_dset]))
        f.close()

        db.add_edge_cost(name_db,
                         collection,
                         voxel_size,
                         volume_offset,
                         chunk,
                         pm_dset)


def solve_candidate_volume(name_db,
                           collection,
                           db_credentials,
                           cc_min_vertices,
                           start_edge_prior,
                           selection_cost,
                           orientation_factor,
                           comb_angle_factor,
                           time_limit,
                           cores,
                           cf_lists,
                           voxel_size,
                           offset=np.array([0.,0.,0.]),
                           mp=True,
                           backend="Gurobi"):

    # Don't forward SIGINT to child processes
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool()
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        for cf_core_ids in cf_lists:
            logging.info("Working on {}".format(cf_core_ids))
                        
            results = []
            if mp:
                for core_id in cf_core_ids:
                    logging.info("Add core {} to pool (mp: {})".format(core_id, mp))
                    results.append(pool.apply_async(solve_core, (cores[core_id],
                                                             name_db,
                                                             collection,
                                                             db_credentials,
                                                             cc_min_vertices,
                                                             start_edge_prior,
                                                             selection_cost,
                                                             orientation_factor,
                                                             comb_angle_factor,
                                                             time_limit,
                                                             voxel_size,
                                                             backend,
                                                             )))
                # Catch exceptions and SIGINTs
                for result in results:
                    result.get(60*60*24*3) 

            else:
                for core_id in cf_core_ids:
                    results.append(solve_core(cores[core_id],
                                                 name_db,
                                                 collection,
                                                 db_credentials,
                                                 cc_min_vertices,
                                                 start_edge_prior,
                                                 selection_cost,
                                                 orientation_factor,
                                                 comb_angle_factor,
                                                 time_limit,
                                                 voxel_size,
                                                 backend,
                                                 ))
    finally:
        pool.terminate()
        pool.join()


def solve_core(core, 
               name_db,
               collection,
               db_credentials,
               cc_min_vertices,
               start_edge_prior,
               selection_cost,
               orientation_factor,
               comb_angle_factor,
               time_limit,
               voxel_size,
               backend):

    try:
        logging.info("Core id {}".format(core.id))
        logging.info("Process core {}...".format(core.id))
        db = DB(db_credentials)
        solver = CoreSolver()
        
        solved = db.is_solved(name_db,
                              collection,
                              core.x_lim_core,
                              core.y_lim_core,
                              core.z_lim_core)

        if not solved:
            g1, index_map = db.get_g1(name_db,
                                      collection,
                                      x_lim=core.x_lim_context,
                                      y_lim=core.y_lim_context,
                                      z_lim=core.z_lim_context)

            solutions = solver.solve_subgraph(g1,
                                              index_map,
                                              cc_min_vertices=cc_min_vertices,
                                              start_edge_prior=start_edge_prior,
                                              selection_cost=selection_cost,
                                              orientation_factor=orientation_factor,
                                              comb_angle_factor=comb_angle_factor,
                                              core_id=core.id,
                                              voxel_size=voxel_size,
                                              time_limit=time_limit,
                                              backend=backend)


            for solution in solutions:
                db.write_solution(solution, 
                                  index_map,
                                  name_db,
                                  collection,
                                  x_lim=core.x_lim_core,
                                  y_lim=core.y_lim_core,
                                  z_lim=core.z_lim_core,
                                  id_writer=core.id)

            db.write_solved(name_db,
                            collection,
                            core)
        else:
            logging.info("Skip core {}, already solved...".format(core.id))

        return core.id
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


if __name__ == "__main__":
    #cfg_dict = read_config("/groups/funke/home/ecksteinn/Projects/microtubules/cremi/experiments/test_runs/run_7/b+_full/config.ini")
    #track(cfg_dict)
    #cfg_dict = read_config("/groups/funke/home/ecksteinn/Projects/stephan/experiments/setup01/config.ini")
    #track(cfg_dict)


    cfg_dict = read_config("/groups/funke/home/ecksteinn/Projects/microtubules/cremi/miccai/grid_search/bestbet_sm/grid/b+/grid_0/config.ini")
    track(cfg_dict)

