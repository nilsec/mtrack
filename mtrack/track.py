from mtrack.cores import CoreSolver, CoreBuilder, VanillaSolver, DB
from mtrack.preprocessing import DirectionType, g1_to_nml, Chunker,\
                                 stack_to_chunks,\
                                 extract_candidates_double, extract_candidates_single
from mtrack.mt_utils import read_config, check_overlap
from mtrack.graphs.g1_graph import G1
try:
    from mtrack.postprocessing import skeletonize
except:
    pass
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
        """
        Check if candidate extraction needs to be performed
        otherwise it is assumed the given db collection holds
        already extracted candidates.
        """
        if config["candidate_extraction_mode"] == "double":

            if config["prob_map_chunks_perp_dir"] == "None":
                """
                Check if chunking of the probability maps needs to
                be performed, otherwise the chunk dir specified is used.
                """

                dir_perp = chunk_prob_map(volume_shape=config["volume_shape"],
                                          max_chunk_shape=config["max_chunk_shape"],
                                          volume_offset=config["volume_offset"],
                                          voxel_size=config["voxel_size"],
                                          prob_map_h5=config["perp_stack_h5"],
                                          output_dir=os.path.join(config["chunk_output_dir"], "perp"))

                dir_par = chunk_prob_map(volume_shape=config["volume_shape"],
                                          max_chunk_shape=config["max_chunk_shape"],
                                          volume_offset=config["volume_offset"],
                                          voxel_size=config["voxel_size"],
                                          prob_map_h5=config["par_stack_h5"],
                                          output_dir=os.path.join(config["chunk_output_dir"], "par"))

                """
                Update the config with the new output dirs for 
                perp and par chunks respectively.
                """
                config["prob_map_chunks_perp_dir"] = dir_perp
                config["prob_map_chunks_par_dir"] = dir_par

                chunks_perp = [os.path.join(config["prob_map_chunks_perp_dir"], f)\
                               for f in os.listdir(config["prob_map_chunks_perp_dir"]) if f.endswith(".h5")]

                chunks_par = [os.path.join(config["prob_map_chunks_par_dir"], f)\
                              for f in os.listdir(config["prob_map_chunks_par_dir"]) if f.endswith(".h5")]

        elif config["candidate_extraction_mode"] == "single":
            if config["prob_map_chunks_single_dir"] == "None":
                dir_single = chunk_prob_map(volume_shape=config["volume_shape"],
                                            max_chunk_shape=config["max_chunk_shape"],
                                            volume_offset=config["volume_offset"],
                                            voxel_size=config["voxel_size"],
                                            prob_map_h5=config["single_stack_h5"],
                                            output_dir=os.path.join(config["chunk_output_dir"], "single"))

                config["prob_map_chunks_single_dir"] = dir_single

                chunks_single = [os.path.join(config["prob_map_chunks_single_dir"], f)\
                                 for f in os.listdir(config["prob_map_chunks_single_dir"]) if f.endswith(".h5")]

        else:
            raise ValueError("Provide candidate extraction mode as 'single' or 'double' mode")

        """
        Extract id, and volume information from chunks
        and compare with ROI
        """
        chunk_limits = {}
        chunk_ids = {}
        roi_chunks = []

        if config["candidate_extraction_mode"] == "double":
            for f_chunk_perp, f_chunk_par in zip(chunks_perp, chunks_par):
                if not os.path.isfile(f_chunk_perp):
                    raise ValueError("{} is not a file".format(f_chunk_perp))

                f = h5py.File(f_chunk_perp, "r")
                attrs_perp = f["exported_data"].attrs.items()
                f.close()

                if not os.path.isfile(f_chunk_par):
                    raise ValueError("{} is not a file".format(f_chunk_par))

                f = h5py.File(f_chunk_par, "r")
                attrs_par = f["exported_data"].attrs.items()
                f.close()
                
                assert(attrs_perp[0][1] == attrs_par[0][1])
     
                """
                We want to process those chunks where all chunk
                limits overlap with the ROI
                """
                chunk_limit = attrs_perp[1][1]
                chunk_id = attrs_perp[0][1]

                chunk_limits[(f_chunk_perp, f_chunk_par)] = chunk_limit 
                chunk_ids[(f_chunk_perp, f_chunk_par)] = chunk_id

                full_ovlp = np.array([False, False, False])
                for i in range(3):
                    full_ovlp[i] = check_overlap(chunk_limit[i], roi[i])

                if np.all(full_ovlp):
                    roi_chunks.append((f_chunk_perp, f_chunk_par))
        else:
            for f_chunk in chunks_single:
                if not os.path.isfile(f_chunk):
                    raise ValueError("{} is not a file".format(f_chunk))

                f = h5py.File(f_chunk, "r")
                attrs = f["exported_data"].attrs.items()
                f.close()

                chunk_limit = attrs[1][1]
                chunk_id = attrs[0][1]

                chunk_limits[f_chunk] = chunk_limit 
                chunk_ids[f_chunk] = chunk_id

                full_ovlp = np.array([False, False, False])
                for i in range(3):
                    full_ovlp[i] = check_overlap(chunk_limit[i], roi[i])

                if np.all(full_ovlp):
                    roi_chunks.append(f_chunk)

        """
        Extract candidates from all ROI chunks and write to specified
        database.
        """

        if config["candidate_extraction_mode"] == "double":
            pm_chunks = [[pm[0] for pm in roi_chunks], [pm[1] for pm in roi_chunks]]
            ps = DirectionType(config["point_threshold_perp"], config["point_threshold_par"])
            gs = DirectionType(config["gaussian_sigma_perp"], config["gaussian_sigma_par"])
        else:
            pm_chunks = roi_chunks
            ps = config["point_threshold_single"]
            gs = config["gaussian_sigma_single"]

        write_candidate_graph(pm_chunks=pm_chunks,
                              mode=config["candidate_extraction_mode"],
                              name_db=config["name_db"],
                              collection=config["name_collection"],
                              db_credentials=config["db_credentials"],
                              gs=gs,
                              ps=ps,
                              distance_threshold=config["distance_threshold"],
                              voxel_size=config["voxel_size"],
                              cores=cores,
                              cf_lists=cf_lists,
                              volume_offset=config["volume_offset"],
                              overwrite=True,
                              mp=config["mp"])

        
        # Clean up chunks
        if config["candidate_extraction_mode"] == "double":
            shutil.rmtree(config["prob_map_chunks_perp_dir"])
            shutil.rmtree(config["prob_map_chunks_par_dir"])
        else:
            shutil.rmtree(config["prob_map_chunks_single_dir"])

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
                               distance_factor=config["distance_factor"],
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
                   output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunker = Chunker(volume_shape,
                      max_chunk_shape,
                      voxel_size,
                      offset=volume_offset)

    chunks = chunker.chunk()

    stack_to_chunks(input_stack=prob_map_h5,
                    output_dir=output_dir,
                    chunks=chunks,
                    volume_offset=volume_offset)

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

def write_candidate_graph(pm_chunks,
                          mode,
                          name_db,
                          collection,
                          db_credentials,
                          gs,
                          ps,
                          distance_threshold,
                          voxel_size,
                          cores,
                          cf_lists,
                          volume_offset,
                          overwrite=False,
                          mp=True):
    """
    This function has two modes:
    
    double: If double is selected pm_chunks
            is expected to contain two lists
            with pm_chunks[0] == perpendicular_chunks
            and  pm_chunks[1] == parallel_chunks.
            This is the option that should be selected
            for a pipeline utilising two seperate
            classifiers for perpendicular and
            parallel microtubules respectively.
            If ilastik is used this is the preferred
            option.

    single: If single is selected pm_chunks
            should contain a single list with 
            the probability map of the combined 
            predictions.
    """

    if not mode in ["single", "double"]:
        raise ValueError("Choose between 'single' or 'double' mode")


    logging.info("Extract pm chunks...")
    db = DB(db_credentials)
    n_chunk = 0
    id_offset = 1
    
    # Overwrite if necesseray:
    graph = db.get_collection(name_db, collection, overwrite=overwrite)


    if mode == "double":
        pm_chunks_perp = pm_chunks[0]
        pm_chunks_par = pm_chunks[1]

        for pm_perp, pm_par in zip(pm_chunks_perp, pm_chunks_par):
            logging.info("Extract chunk {}/{}...".format(n_chunk, len(pm_chunks_par)))

            prob_map_stack = DirectionType(pm_perp, pm_par)

            f = h5py.File(pm_perp, "r")
            attrs = f["exported_data"].attrs.items()
            f.close()

            chunk_limits = attrs[1][1]
            offset_chunk = [chunk_limits[0][0], 
                            chunk_limits[1][0], 
                            chunk_limits[2][0]]

            print offset_chunk

            candidates = extract_candidates_double(prob_map_stack,
                                                   gs,
                                                   ps,
                                                   voxel_size,
                                                   bounding_box=None,
                                                   bs_output_dir=None,
                                                   offset_pos=offset_chunk,
                                                   identifier_0=id_offset)

            id_offset_tmp = db.write_candidates(name_db,
                                                collection,
                                                candidates,
                                                voxel_size,
                                                overwrite=False)


            assert(graph.find({"selected": {"$exists": True}}).count() == id_offset_tmp)

            id_offset = id_offset_tmp + 1
            n_chunk += 1

    else:
        for chunk in pm_chunks:
            logging.info("Extract chunk {}/{}...".format(n_chunk, len(pm_chunks)))

            f = h5py.File(chunk, "r")
            attrs = f["exported_data"].attrs.items()
            f.close()

            chunk_limits = attrs[1][1]
            offset_chunk = [chunk_limits[0][0], 
                            chunk_limits[1][0], 
                            chunk_limits[2][0]]

            candidates = extract_candidates_single(chunk,
                                                   gs,
                                                   ps,
                                                   voxel_size,
                                                   binary_closing=True,
                                                   offset_pos=offset_chunk,
                                                   identifier_0=id_offset)

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


def solve_candidate_volume(name_db,
                           collection,
                           db_credentials,
                           cc_min_vertices,
                           start_edge_prior,
                           selection_cost,
                           orientation_factor,
                           distance_factor,
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
                                                             distance_factor,
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
                                                 distance_factor,
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
               distance_factor,
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
                                              distance_factor=distance_factor,
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


def cluster(name_db,
            collection,
            db_credentials,
            roi,
            output_dir,
            epsilon_lines,
            epsilon_volumes,
            min_overlap_volumes,
            cluster_orientation_factor,
            remove_singletons,
            voxel_size,
            use_ori=True):

    db = DB(db_credentials)

    x_lim_roi = {"min": roi[0][0] * voxel_size[0],
                 "max": roi[0][1] * voxel_size[0]}
    y_lim_roi = {"min": roi[1][0] * voxel_size[1],
                 "max": roi[1][1] * voxel_size[1]}
    z_lim_roi = {"min": roi[2][0] * voxel_size[2],
                 "max": roi[2][1] * voxel_size[2]}

    g1, index_map = db.get_selected(name_db,
                                    collection,
                                    x_lim=x_lim_roi,
                                    y_lim=y_lim_roi,
                                    z_lim=z_lim_roi)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    solution_file = os.path.join(output_dir, "cluster_roi.gt")
    g1.save(solution_file)

    skeletonize(solution_file,
                output_dir,
                epsilon_lines,
                epsilon_volumes,
                min_overlap_volumes,
                canvas_shape=[roi[2][1] - roi[2][0], 
                              roi[1][1] - roi[1][0], 
                              roi[0][1] - roi[0][0]],
                offset=np.array([roi[0][0],
                                 roi[1][0],
                                 roi[2][0]]),
                orientation_factor=cluster_orientation_factor,
                remove_singletons=remove_singletons,
                use_ori=use_ori,
                voxel_size=voxel_size)


if __name__ == "__main__":
    #cfg_dict = read_config("/groups/funke/home/ecksteinn/Projects/microtubules/cremi/experiments/test_runs/run_7/b+_full/config.ini")
    #track(cfg_dict)
    cfg_dict = read_config("/groups/funke/home/ecksteinn/Projects/microtubules/cremi/miccai/grid_search/baseline/grid/b+/grid_0/config.ini")
    track(cfg_dict)
