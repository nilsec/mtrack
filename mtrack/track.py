from mtrack.cores import CoreSolver, CoreBuilder, CoreScheduler, ExceptionWrapper, VanillaSolver
from mtrack.preprocessing import DirectionType, g1_to_nml, Chunker, slices_to_chunks, get_prob_map_ilastik
from mtrack.mt_utils import read_config, check_overlap
from mtrack.postprocessing import skeletonize
from solve import solve

import numpy as np
import h5py
import os
import multiprocessing, logging
import sys


def chunk_pms(volume_shape,
              max_chunk_shape,
              voxel_size,
              overlap,
              prob_map_perp_dir,
              prob_map_par_dir,
              output_dir):

    dir_perp = os.path.join(output_dir, "perp")
    dir_par = os.path.join(output_dir, "par") 

    if not os.path.exists(dir_par):
        os.makedirs(dir_par)

    if not os.path.exists(dir_perp):
        os.makedirs(dir_perp)
    
    chunker = Chunker(volume_shape,
                      max_chunk_shape,
                      voxel_size,
                      overlap)

    chunks = chunker.chunk()
    slices_to_chunks(prob_map_perp_dir,
                     dir_perp,
                     chunks)

    slices_to_chunks(prob_map_par_dir,
                     dir_par,
                     chunks)

    return dir_perp, dir_par


def cluster(name_db,
            collection,
            roi,
            output_dir,
            epsilon_lines,
            epsilon_volumes,
            min_overlap_volumes,
            cluster_orientation_factor,
            remove_singletons,
            voxel_size,
            use_ori=True):

    solver = CoreSolver()
    client = solver._get_client(name_db, collection, overwrite=False)

    x_lim_roi = {"min": roi[0][0] * voxel_size[0],
                 "max": roi[0][1] * voxel_size[0]}
    y_lim_roi = {"min": roi[1][0] * voxel_size[1],
                 "max": roi[1][1] * voxel_size[1]}
    z_lim_roi = {"min": roi[2][0] * voxel_size[2],
                 "max": roi[2][1] * voxel_size[2]}

    vertices, edges = solver.get_subgraph(name_db,
                                          collection,
                                          x_lim=x_lim_roi,
                                          y_lim=y_lim_roi,
                                          z_lim=z_lim_roi)

    g1, index_map = solver.subgraph_to_g1(vertices, 
                                          edges)

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


def extract_pm_chunks(pm_chunks_par,
                      pm_chunks_perp,
                      name_db,
                      collection,
                      gs,
                      ps,
                      distance_threshold,
                      voxel_size,
                      overwrite=False):


    
    solver = CoreSolver()

    print "Extract pm chunks..."

    n_chunk = 0
    id_offset = 0

    """    
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    id_offset = manager.Value("i", 0, lock=lock)
    pool = multiprocessing.Pool()
    """

    if overwrite:
        solver._get_client(name_db, collection, overwrite=overwrite)

    for pm_perp, pm_par in zip(pm_chunks_perp, pm_chunks_par):
        print "Extract chunk {}/{}...".format(n_chunk, len(pm_chunks_par))

        prob_map_stack = DirectionType(pm_perp, pm_par)

        f = h5py.File(pm_perp, "r")
        attrs = f["exported_data"].attrs.items()
        f.close()

        chunk_limits = attrs[1][1]
        offset_chunk = [chunk_limits[0][0], 
                        chunk_limits[1][0], 
                        chunk_limits[2][0]]

        id_offset_tmp = solver.save_candidates(name_db,
                                           prob_map_stack,
                                           offset_chunk,
                                           gs,
                                           ps,
                                           voxel_size,
                                           id_offset=id_offset,
                                           collection=collection,
                                           overwrite=False)
        id_offset = id_offset_tmp
        """
        with id_offset.get_lock():
            id_offset = id_offset_tmp
        """
        n_chunk += 1


def solve_candidate_volume(name_db,
                           collection,
                           distance_threshold,
                           cc_min_vertices,
                           start_edge_prior,
                           selection_cost,
                           distance_factor,
                           orientation_factor,
                           comb_angle_factor,
                           time_limit,
                           hcs,
                           core_size,
                           context_size,
                           volume_size,
                           min_core_overlap,
                           voxel_size,
                           output_dir,
                           save_candidates,
                           save_cores,
                           save_connected,
                           save_core_graphs,
                           gt,
                           nml,
                           x_lim_out,
                           y_lim_out,
                           z_lim_out,
                           copy_candidates=True,
                           copy_target="microtubules",
                           offset=np.array([0.,0.,0.]),
                           overwrite_copy_target=False,
                           skip_solved_cores=True,
                           mp=True):

    solver = CoreSolver()

    if copy_candidates:
        pipeline = [{"$match": {}},
                    {"$out": copy_target},]

        db = solver._get_db(name_db)
        if copy_target not in db.collection_names():
            print "{} does not exist, copy candidate collection...".format(copy_target)
            db["candidates"].aggregate(pipeline)
            collection = copy_target

        else:
            if overwrite_copy_target:
                print "Overwrite " + copy_target + "..."
                db[copy_target].remove({})
                assert(db[copy_target].find({}).count() == 0)
                db["candidates"].aggregate(pipeline)
                collection = copy_target
            else:
                collection = copy_target
 
    builder = CoreBuilder(volume_size,
                          core_size,
                          context_size,
                          min_core_overlap,
                          offset)

    ovlp = builder._get_overlap()[1]
    assert(np.all(np.array(ovlp) == 0))
    cores = builder.generate_cores()

    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool()
    core_queue = manager.Queue()
    lock = manager.Lock()

    scheduler = CoreScheduler(cores)
    
    cores_finished = manager.list()
    cores_active = manager.list()
    cores_pending = manager.list()
    
    mpl = multiprocessing.log_to_stderr()
    mpl.setLevel(logging.DEBUG)
    
    """
    Populate the queue with all 
    available cores that can 
    be processed initially and independently
    """
    cores_available = True
    while cores_available:
        print "Core available: {}".format(cores_available)
        core = scheduler.request_core()
        if core is not None:
            core_queue.put(core)
            cores_pending.append(core.id)
        else:
            cores_available=False

    processed_cores = 0
    if mp: 
        while processed_cores < len(cores):
            print "Apply async..."
            print "Cores processing", len(cores_active) + len(cores_pending)
            print "Finished/cores", len(cores_finished), len(cores)
            core = core_queue.get(block=True)
            handler = pool.apply_async(solve_core, 
                                        (
                                          core, 
                                          cores,
                                          core_queue, 
                                          cores_active, 
                                          cores_pending,
                                          cores_finished,
                                          lock,
                                          solver,
                                          name_db,
                                          collection,
                                          distance_threshold,
                                          cc_min_vertices,
                                          start_edge_prior,
                                          selection_cost,
                                          distance_factor,
                                          orientation_factor,
                                          comb_angle_factor,
                                          time_limit,
                                          hcs,
                                          core_size,
                                          context_size,
                                          volume_size,
                                          min_core_overlap,
                                          voxel_size,
                                          output_dir,
                                          save_candidates,
                                          save_cores,
                                          save_connected,
                                          save_core_graphs,
                                          gt,
                                          nml,
                                          x_lim_out,
                                          y_lim_out,
                                          z_lim_out,
                                          mp,
                                          copy_candidates,
                                          copy_target,
                                          offset,
                                          overwrite_copy_target,
                                          skip_solved_cores),
                                          callback=ExceptionWrapper.exception_handler
                                        )
            processed_cores += 1
     
        pool.close()
        pool.join()

    else:
        while processed_cores < len(cores):
            print "Solve core (SP)..."
            print "Cores processing", len(cores_active) + len(cores_pending)
            print "Finished/cores", len(cores_finished), len(cores)
            
            core = core_queue.get(block=True)

            solve_core(
                        core, 
                        cores,
                        core_queue, 
                        cores_active, 
                        cores_pending,
                        cores_finished,
                        lock,
                        solver,
                        name_db,
                        collection,
                        distance_threshold,
                        cc_min_vertices,
                        start_edge_prior,
                        selection_cost,
                        distance_factor,
                        orientation_factor,
                        comb_angle_factor,
                        time_limit,
                        hcs,
                        core_size,
                        context_size,
                        volume_size,
                        min_core_overlap,
                        voxel_size,
                        output_dir,
                        save_candidates,
                        save_cores,
                        save_connected,
                        save_core_graphs,
                        gt,
                        nml,
                        x_lim_out,
                        y_lim_out,
                        z_lim_out,
                        mp,
                        copy_candidates,
                        copy_target,
                        offset,
                        overwrite_copy_target,
                        skip_solved_cores
                        )
            processed_cores += 1


def solve_core(core, 
               cores,
               core_queue,
               cores_active,
               cores_pending,
               cores_finished,
               lock,
               solver,
               name_db,
               collection,
               distance_threshold,
               cc_min_vertices,
               start_edge_prior,
               selection_cost,
               distance_factor,
               orientation_factor,
               comb_angle_factor,
               time_limit,
               hcs,
               core_size,
               context_size,
               volume_size,
               min_core_overlap,
               voxel_size,
               output_dir,
               save_candidates,
               save_cores,
               save_connected,
               save_core_graphs,
               gt,
               nml,
               x_lim_out,
               y_lim_out,
               z_lim_out,
               mp,
               copy_candidates=True,
               copy_target="microtubules",
               offset=np.array([0.,0.,0.]),
               overwrite_copy_target=False,
               skip_solved_cores=True):
    
    try:
        lock.acquire()
        cores_pending.remove(core.id)
        cores_active.append(core.id)
        lock.release()

        print "Core id {}".format(core.id)
        print "Process core {}...".format(core.id)
        core_finished = False

        if skip_solved_cores:
            if solver.core_finished(core_id=core.id,
                                    name_db=name_db,
                                    collection="candidates") or\
                solver.core_finished(core_id=core.id,
                                     name_db=name_db,
                                     collection=collection):
                print "Core already solved... continue"
                core_finished = True

        if not core_finished:
            vertices, edges = solver.get_subgraph(name_db,
                                                  collection,
                                                  x_lim=core.x_lim_context,
                                                  y_lim=core.y_lim_context,
                                                  z_lim=core.z_lim_context)

            if save_candidates:
                g1, index_map = solver.subgraph_to_g1(vertices, 
                                                      edges)
                if gt:
                    candidates_core_dir = os.path.join(output_dir, "cores/candidates")

                    if not os.path.exists(candidates_core_dir):
                        os.makedirs(candidates_core_dir)
                
                    g1.save(os.path.join(candidates_core_dir, 
                                        "core_{}.gt".format(core.id)))
                if nml:
                    g1_to_nml(g1, 
                              os.path.join(candidates_core_dir, "core_{}.nml".format(core.id)), 
                              knossos=True, 
                              voxel_size=voxel_size)
     
            solutions = solver.solve_subgraph(g1,
                                              index_map,
                                              distance_threshold=distance_threshold,
                                              cc_min_vertices=cc_min_vertices,
                                              start_edge_prior=start_edge_prior,
                                              selection_cost=selection_cost,
                                              distance_factor=distance_factor,
                                              orientation_factor=orientation_factor,
                                              comb_angle_factor=comb_angle_factor,
                                              core_id=core.id,
                                              save_connected=save_connected,
                                              output_dir=output_dir,
                                              voxel_size=voxel_size,
                                              time_limit=time_limit,
                                              hcs=hcs)

            if save_core_graphs:
                core_graph_dir = os.path.join(output_dir, "core_graphs/core_{}".format(core.id))
                if not os.path.exists(core_graph_dir):
                    os.makedirs(core_graph_dir)
                    for j in range(len(solutions)):
                        g1_to_nml(solutions[j],
                                  os.path.join(core_graph_dir, "cc_{}.nml".format(j)),
                                  knossos=True,
                                  voxel_size=voxel_size)

                        g1.save(os.path.join(core_graph_dir, "cc_{}.gt".format(j)))

            for solution in solutions:
                solver.write_solution(solution, 
                                      index_map,
                                      name_db,
                                      collection,
                                      x_lim=core.x_lim_core,
                                      y_lim=core.y_lim_core,
                                      z_lim=core.z_lim_core)

            solver.remove_deg_0_vertices(name_db,
                                         collection,
                                         x_lim=core.x_lim_core,
                                         y_lim=core.y_lim_core,
                                         z_lim=core.z_lim_core)
            
            solver.finish_core(core_id=core.id,
                               name_db=name_db,
                               collection="candidates")

        if save_cores:
            vertices, edges = solver.get_subgraph(name_db,
                                                  collection,
                                                  x_lim=x_lim_out,
                                                  y_lim=y_lim_out,
                                                  z_lim=z_lim_out)

            g1, index_map = solver.subgraph_to_g1(vertices, edges, set_partner=False)

            if gt:
                gt_core_dir = os.path.join(os.path.join(output_dir, "cores/gt"))

                if not os.path.exists(gt_core_dir):
                    os.makedirs(gt_core_dir)

                g1.save(os.path.join(gt_core_dir, "core_{}.gt".format(core.id)))

            if nml:
                nml_core_dir = os.path.join(os.path.join(output_dir, "cores/nml"))
                    
                if not os.path.exists(nml_core_dir):
                    os.makedirs(nml_core_dir)

                g1_to_nml(g1, 
                          os.path.join(nml_core_dir, "core{}.nml".format(core.id)), 
                          knossos=True, 
                          voxel_size=voxel_size)

        lock.acquire()    
        cores_active.remove(core.id)
        cores_finished.append(core.id)
        print "Extend queue..."

        for new_core in cores:
            if not (new_core.id in cores_finished):
                if not (new_core.id in cores_active):
                    if not (new_core.id in cores_pending):
                        new_core_nbs = new_core.nbs

                        print "new_core_id", new_core.id
                        print "nbs", new_core.nbs
                        print "active", list(cores_active)
                        print "finished", list(cores_finished)

                        if not (set(cores_active) & set(new_core_nbs)):
                            print "Add core {}".format(new_core.id)

                            core_queue.put(new_core, block=True)
                            cores_pending.append(new_core.id)
        
        lock.release()
        print "finished", list(cores_finished)

    except Exception as e:
        if mp:
            return ExceptionWrapper(e)
        else:
            raise e

def clean_up(name_db, 
             collection, 
             x_lim, 
             y_lim, 
             z_lim, 
             save_roi=True, 
             nml=True, 
             gt=False,
             output_dir=None,
             voxel_size=None):

    solver = CoreSolver()

    solver.remove_deg_0_vertices(name_db,
                                 collection,
                                 x_lim=x_lim,
                                 y_lim=y_lim,
                                 z_lim=z_lim)


    if save_roi:
        vertices, edges = solver.get_subgraph(name_db,
                                              collection,
                                              x_lim=x_lim,
                                              y_lim=y_lim,
                                              z_lim=z_lim)

        g1, index_map = solver.subgraph_to_g1(vertices, 
                                              edges, 
                                              set_partner=False)

        if gt:
            g1.save(os.path.join(output_dir, "roi.gt"))
    
        if nml:
            g1_to_nml(g1, 
                      os.path.join(output_dir, "roi.nml"), 
                      knossos=True,
                      voxel_size=voxel_size)


def track(config_path):
    config = read_config(config_path)
    roi = [config["roi_x"], config["roi_y"], config["roi_z"]]

    if config["extract_prob_maps"]:
        """
        Extract probability map via elastic classifier from specified input dir and ilastik project.
        """
        get_prob_map_ilastik(config["image_dir"],
                             config["pm_output_dir_perp"],
                             config["ilastik_source_dir"],
                             config["ilastik_project_perp"],
                             verbose=True,
                             file_extension=config["file_extension"])
        
        get_prob_map_ilastik(config["image_dir"],
                             config["pm_output_dir_par"],
                             config["ilastik_source_dir"],
                             config["ilastik_project_par"],
                             verbose=True,
                             file_extension=config["file_extension"])

    if config["core_wise"]:
        if config["solve"]:

            if config["extract_candidates"]:
                """
                Check if candidate extraction needs to be performed
                otherwise it is assumed the given db collection holds
                already extracted candidates.
                """

                if config["prob_map_chunks_perp_dir"] == "None":
                    """
                    Check if chunking of the probability maps needs to
                    be performed, otherwise the chunk dir specified is used.
                    """

                    dir_perp, dir_par = chunk_pms(volume_shape=config["volume_shape"],
                                                  max_chunk_shape=config["max_chunk_shape"],
                                                  voxel_size=config["voxel_size"],
                                                  overlap=config["chunk_overlap"],
                                                  prob_map_perp_dir=config["prob_maps_perp_dir"],
                                                  prob_map_par_dir=config["prob_maps_par_dir"],
                                                  output_dir=config["chunk_output_dir"])

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

                """
                Check which Chunks hold the ROI.
                """
                

                """
                Extract id, and volume information from chunks
                and compare with ROI
                """
                chunk_limits = {}
                chunk_ids = {}
                roi_chunks = []

                for f_chunk_perp, f_chunk_par in zip(chunks_perp, chunks_par):
                    f = h5py.File(f_chunk_perp, "r")
                    attrs_perp = f["exported_data"].attrs.items()
                    f.close()

                    f = h5py.File(f_chunk_par, "r")
                    attrs_par = f["exported_data"].attrs.items()
                    f.close()

                    # Sanity check that par and perp chunks have same id
                    assert(attrs_perp[0][1] == attrs_par[0][1])
         
                    """
                    We want to process those changes where all chunk
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

                """
                Extract candidates from all ROI chunks and write to specified
                database. The collection defaults to /candidates/.
                """
                extract_pm_chunks(pm_chunks_par=[pm[1] for pm in roi_chunks],
                                  pm_chunks_perp=[pm[0] for pm in roi_chunks],
                                  name_db=config["db_name"],
                                  collection="candidates",
                                  gs=DirectionType(config["gaussian_sigma_perp"], 
                                                   config["gaussian_sigma_par"]),
                                  ps=DirectionType(config["point_threshold_perp"],
                                                   config["point_threshold_par"]),
                                  distance_threshold=config["distance_threshold"],
                                  voxel_size=config["voxel_size"],
                                  overwrite=config["overwrite_candidates"])

            # Update roi volume size:
            roi_volume_size = np.array([r[1] - r[0] for r in roi]) * config["voxel_size"]
            roi_offset = np.array([r[0] for r in roi]) * config["voxel_size"]

            """
            Solve the ROI and write to specified database. The result
            is written out depending on the options in the Output section
            of the config file. The collection defaults to /microtubules/.
            """

            x_lim_roi = {"min": roi_offset[0],
                         "max": roi_offset[0] + roi_volume_size[0]}
            y_lim_roi = {"min": roi_offset[1],
                         "max": roi_offset[1] + roi_volume_size[1]}
            z_lim_roi = {"min": roi_offset[2],
                         "max": roi_offset[2] + roi_volume_size[2]}


            solve_candidate_volume(name_db=config["db_name"],
                                   collection="candidates",
                                   distance_threshold=config["distance_threshold"],
                                   cc_min_vertices=config["cc_min_vertices"],
                                   start_edge_prior=config["start_edge_prior"],
                                   selection_cost=config["selection_cost"],
                                   distance_factor=config["distance_factor"],
                                   orientation_factor=config["orientation_factor"],
                                   comb_angle_factor=config["comb_angle_factor"],
                                   time_limit=config["time_limit_per_cc"],
                                   hcs=config["get_hcs"],
                                   core_size=config["core_size"] * config["voxel_size"],
                                   context_size=config["context_size"] * config["voxel_size"],
                                   volume_size=roi_volume_size,
                                   min_core_overlap=config["min_core_overlap"] * config["voxel_size"],
                                   voxel_size=config["voxel_size"],
                                   output_dir=config["output_dir"],
                                   save_candidates=config["save_candidates"],
                                   save_cores=config["save_cores"],
                                   save_connected=config["save_connected"],
                                   save_core_graphs=config["save_core_graphs"],
                                   gt=config["gt"],
                                   nml=config["nml"],
                                   x_lim_out=x_lim_roi,
                                   y_lim_out=y_lim_roi,
                                   z_lim_out=z_lim_roi,
                                   copy_candidates=True,
                                   copy_target="microtubules",
                                   offset=np.array(roi_offset),
                                   overwrite_copy_target=config["overwrite_copy_target"],
                                   skip_solved_cores=config["skip_solved_cores"],
                                   mp=config["mp"]) 

            """
            Clean up all remaining degree 0 vertices in context area inside
            the solved ROI.
            """
            clean_up(name_db=config["db_name"], 
                     collection="microtubules", 
                     x_lim=x_lim_roi, 
                     y_lim=y_lim_roi, 
                     z_lim=z_lim_roi, 
                     save_roi=config["save_roi"], 
                     nml=config["nml"], 
                     gt=config["gt"],
                     output_dir=config["output_dir"],
                     voxel_size=config["voxel_size"]) 

        if config["cluster"]:
           cluster(name_db=config["db_name"],
                   collection="microtubules",
                   roi=roi,
                   output_dir=config["cluster_output_dir"],
                   epsilon_lines=config["epsilon_lines"],
                   epsilon_volumes=config["epsilon_volumes"],
                   min_overlap_volumes=config["min_overlap_volumes"],
                   cluster_orientation_factor=config["cluster_orientation_factor"],
                   remove_singletons=config["remove_singletons"],
                   voxel_size=config["voxel_size"],
                   use_ori=config["use_ori"])

    else:
        if config["solve"]:
            if config["prob_map_chunks_perp_dir"] == "None":
                """
                Check if chunking of the probability maps needs to
                be performed, otherwise the chunk dir specified is used.
                """

                dir_perp, dir_par = chunk_pms(volume_shape=config["volume_shape"],
                                              max_chunk_shape=config["max_chunk_shape"],
                                              voxel_size=config["voxel_size"],
                                              overlap=config["chunk_overlap"],
                                              prob_map_perp_dir=config["prob_maps_perp_dir"],
                                              prob_map_par_dir=config["prob_maps_par_dir"],
                                              output_dir=config["chunk_output_dir"])

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

        
            """
            Extract id, and volume information from chunks
            and compare with ROI
            """
            chunk_limits = {}
            chunk_ids = {}
            roi_chunks = []

            for f_chunk_perp, f_chunk_par in zip(chunks_perp, chunks_par):
                f = h5py.File(f_chunk_perp, "r")
                attrs_perp = f["exported_data"].attrs.items()
                f.close()

                f = h5py.File(f_chunk_par, "r")
                attrs_par = f["exported_data"].attrs.items()
                f.close()

                # Sanity check that par and perp chunks have same id
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

            """
            Extract candidates from all ROI chunks and write to specified
            database. The collection defaults to /candidates/.
            """
            solver = VanillaSolver()

            """
            Extract all candidates from each chunk:
            """
            candidates = []
            for chunk in roi_chunks:
                pm_stack_chunk = DirectionType(chunk[0], chunk[1])

                f = h5py.File(chunk[0], "r")
                attrs = f["exported_data"].attrs.items()
                f.close()

                chunk_limits = attrs[1][1]
                offset_chunk = [chunk_limits[0][0], 
                                chunk_limits[1][0], 
                                chunk_limits[2][0]]

        
                candidates += solver.get_candidates(prob_map_stack_chunk=pm_stack_chunk,
                                                    offset_chunk=offset_chunk,
                                                    gs=DirectionType(config["gaussian_sigma_perp"],
                                                                         config["gaussian_sigma_par"]),
                                                    ps=DirectionType(config["point_threshold_perp"],
                                                                         config["point_threshold_par"]),
                                                    voxel_size=config["voxel_size"],
                                                    id_offset=len(candidates))

            g1 = solver.get_g1_graph(candidates,
                                     voxel_size=config["voxel_size"])

            g1_connected = solver.connect_g1_graph(g1,
                                                   distance_threshold=config["distance_threshold"],
                                                   output_dir=config["output_dir"],
                                                   voxel_size=config["voxel_size"])

            cc_solutions = solver.solve_g1_graph(g1_connected,
                                                 cc_min_vertices=config["cc_min_vertices"],
                                                 start_edge_prior=config["start_edge_prior"],
                                                 selection_cost=config["selection_cost"],
                                                 distance_factor=config["distance_factor"],
                                                 orientation_factor=config["orientation_factor"],
                                                 comb_angle_factor=config["comb_angle_factor"],
                                                 output_dir=config["output_dir"],
                                                 time_limit=config["time_limit_per_cc"],
                                                 voxel_size=config["voxel_size"])

            solver.save_solutions(solutions=cc_solutions,
                                  voxel_size=config["voxel_size"],
                                  output_dir=config["output_dir"])
                
if __name__ == "__main__":
    track("../config.ini")
