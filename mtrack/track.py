from mtrack.cores import CoreSolver, CoreBuilder, VanillaSolver
from mtrack.preprocessing import DirectionType, g1_to_nml, Chunker,\
                                 stack_to_chunks, ilastik_get_prob_map
from mtrack.mt_utils import read_config, check_overlap
from mtrack.postprocessing import skeletonize
from mtrack.evaluation import evaluate
from solve import solve

import numpy as np
import h5py
import os
import multiprocessing, logging
import sys
import signal
import traceback


def chunk_pms(volume_shape,
              max_chunk_shape,
              voxel_size,
              overlap,
              perp_stack_h5,
              par_stack_h5,
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
    
    stack_to_chunks(input_stack=perp_stack_h5,
                    output_dir=dir_perp,
                    chunks=chunks)
    stack_to_chunks(input_stack=par_stack_h5,
                    output_dir=dir_par,
                    chunks=chunks)

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
                           voxel_size,
                           copy_target="microtubules",
                           offset=np.array([0.,0.,0.]),
                           overwrite_copy_target=False,
                           skip_solved_cores=True,
                           mp=True):

    solver = CoreSolver()

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
                          offset)
    
    # Generate core geometry
    cores = builder.generate_cores()

    # Get conflict free core lists
    cf_lists = builder.gen_cfs()

    # Don't forward SIGINT to child processes
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool()
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        for cf_core_ids in cf_lists:
            print "Working on ", cf_core_ids
            
            results = []
            for core_id in cf_core_ids:
                if mp:
                    results.append(pool.apply_async(solve_core, (cores[core_id],
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
                                                             voxel_size,
                                                             skip_solved_cores,
                                                             )))
                    # Catch exceptions and SIGINTs
                    for result in results:
                        result.get(60*60*24*3) 

                else:
                    results.append(solve_core(cores[core_id],
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
                                                             voxel_size,
                                                             skip_solved_cores,
                                                             ))

    finally:
        pool.terminate()
        pool.join()


def solve_core(core, 
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
               voxel_size,
               skip_solved_cores):

    try:
        print "Core id {}".format(core.id)
        print "Process core {}...".format(core.id)

        core_finished = False
        if skip_solved_cores:
            if solver.core_finished(core_id=core.id,
                                    name_db=name_db,
                                    collection="candidates"):

                print "Core already solved... continue"
                core_finished = True

        if not core_finished:
            vertices, edges = solver.get_subgraph(name_db,
                                                  collection,
                                                  x_lim=core.x_lim_context,
                                                  y_lim=core.y_lim_context,
                                                  z_lim=core.z_lim_context)

            g1, index_map = solver.subgraph_to_g1(vertices,
                                                  edges)

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
                                              voxel_size=voxel_size,
                                              time_limit=time_limit,
                                              hcs=hcs)


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

        return core.id
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def clean_up(name_db, 
             collection, 
             x_lim, 
             y_lim, 
             z_lim):

    solver = CoreSolver()

    solver.remove_deg_0_vertices(name_db,
                                 collection,
                                 x_lim=x_lim,
                                 y_lim=y_lim,
                                 z_lim=z_lim)


def evaluate_roi(name_db,
                 collection,
                 x_lim,
                 y_lim,
                 z_lim,
                 tracing_file,
                 chunk_size,
                 distance_tolerance,
                 dummy_cost,
                 edge_selection_cost,
                 pair_cost_factor,
                 max_edges,
                 voxel_size,
                 time_limit,
                 output_dir):

    solver = CoreSolver()

    vertices, edges = solver.get_subgraph(name_db,
                                          collection,
                                          x_lim=x_lim,
                                          y_lim=y_lim,
                                          z_lim=z_lim)

    g1, index_map = solver.subgraph_to_g1(vertices,
                                          edges,
                                          set_partner=False)

    evaluate(tracing_file=tracing_file,
             solution_file=g1,
             chunk_size=chunk_size,
             distance_tolerance=distance_tolerance,
             dummy_cost=dummy_cost,
             edge_selection_cost=edge_selection_cost,
             pair_cost_factor=pair_cost_factor,
             max_edges=max_edges,
             voxel_size=voxel_size,
             output_dir=output_dir,
             time_limit=time_limit,
             tracing_line_paths=None,
             rec_line_paths=None)


def track(config_path):
    config = read_config(config_path)
    roi = [config["roi_x"], config["roi_y"], config["roi_z"]]

    roi_volume_size = np.array([r[1] - r[0] for r in roi]) * config["voxel_size"]
    roi_offset = np.array([r[0] for r in roi]) * config["voxel_size"]

    x_lim_roi = {"min": roi_offset[0],
                 "max": roi_offset[0] + roi_volume_size[0]}

    y_lim_roi = {"min": roi_offset[1],
                 "max": roi_offset[1] + roi_volume_size[1]}

    z_lim_roi = {"min": roi_offset[2],
                 "max": roi_offset[2] + roi_volume_size[2]}

    """
    Extract probability map via Ilastik classifier from specified input dir and ilastik project.
    """
    if config["extract_perp"]:
        perp_stack_h5 = ilastik_get_prob_map(raw=config["raw"],
                                             output_dir=config["pm_output_dir_perp"],
                                             ilastik_source_dir=config["ilastik_source_dir"],
                                             ilastik_project=config["ilastik_project_perp"],
                                             file_extension=config["file_extension"],
                                             h5_dset=config["h5_dset"],
                                             label=config["label"])

        config["perp_stack_h5"] = perp_stack_h5

    if config["extract_par"]: 
        par_stack_h5 = ilastik_get_prob_map(raw=config["raw"],
                                            output_dir=config["pm_output_dir_par"],
                                            ilastik_source_dir=config["ilastik_source_dir"],
                                            ilastik_project=config["ilastik_project_par"],
                                            file_extension=config["file_extension"],
                                            h5_dset=config["h5_dset"],
                                            label=config["label"])
        
        config["par_stack_h5"] = par_stack_h5

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
                                              perp_stack_h5=config["perp_stack_h5"],
                                              par_stack_h5=config["par_stack_h5"],
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

        """
        Solve the ROI and write to specified database. The result
        is written out depending on the options in the Output section
        of the config file. The collection defaults to /microtubules/.
        """

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
                               voxel_size=config["voxel_size"],
                               copy_target="microtubules",
                               offset=np.array(roi_offset),
                               overwrite_copy_target=config["overwrite_copy_target"],
                               skip_solved_cores=config["skip_solved_cores"]) 

        """
        Clean up all remaining degree 0 vertices in context area inside
        the solved ROI.
        """
        clean_up(name_db=config["db_name"], 
                 collection="microtubules", 
                 x_lim=x_lim_roi, 
                 y_lim=y_lim_roi, 
                 z_lim=z_lim_roi)

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


    if config["evaluate"]:
        evaluate_roi(name_db=config["db_name"],
                     collection="microtubules",
                     x_lim=x_lim_roi,
                     y_lim=y_lim_roi,
                     z_lim=z_lim_roi,
                     tracing_file=config["tracing_file"],
                     chunk_size=config["eval_chunk_size"],
                     distance_tolerance=config["eval_distance_tolerance"],
                     dummy_cost=config["eval_dummy_cost"],
                     edge_selection_cost=config["eval_edge_selection_cost"],
                     pair_cost_factor=config["eval_pair_cost_factor"],
                     max_edges=config["max_edges"],
                     voxel_size=config["voxel_size"],
                     output_dir=config["eval_output_dir"],
                     time_limit=config["eval_time_limit"])
 
                
if __name__ == "__main__":
    track("/media/nilsec/d0/gt_mt_data/mtrack/grid_A+/grid_9/config.ini")
