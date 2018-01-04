from mtrack.solve import CoreSolver, CoreBuilder, CoreScheduler, solve
from mtrack.preprocessing import DirectionType, g1_to_nml, Chunker, slices_to_chunks
from mtrack.mt_utils import read_config, check_overlap

import numpy as np
import h5py
import pdb
import os

def chunk(volume_shape,
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

    for pm_perp, pm_par in zip(pm_chunks_perp, pm_chunks_par):
        print "Extract chunk {}/{}...".format(n_chunk, len(pm_chunks_par))
        overwrite_tmp = False

        prob_map_stack = DirectionType(pm_perp, pm_par)

        f = h5py.File(pm_perp, "r")
        attrs = f["exported_data"].attrs.items()
        f.close()

        chunk_limits = attrs[1][1]
        offset_chunk = [chunk_limits[0][0], 
                        chunk_limits[1][0], 
                        chunk_limits[2][0]]

        if overwrite:
            if n_chunk == 0:
                overwrite_tmp=True
            
        id_offset = solver.save_candidates(name_db,
                                           prob_map_stack,
                                           offset_chunk,
                                           gs,
                                           ps,
                                           voxel_size,
                                           id_offset=id_offset,
                                           collection=collection,
                                           overwrite=overwrite_tmp)
        n_chunk += 1


def solve_candidate_volume(name_db,
                           collection,
                           core_size,
                           context_size,
                           volume_size,
                           min_core_overlap,
                           copy_candidates=True,
                           copy_target="microtubules"
                           offset=np.array([0.,0.,0.])):

    solver = CoreSolver()

    if copy_candidates:
        print "Copy candidate collection..."

        pipeline = [{"$match": {}},
                    {"$out": copy_target},]

        db = solver._get_db(name_db)
        db.source_collection.aggregate(pipeline)
        collection = copy_target
    
        
    builder = CoreBuilder(volume_size,
                          core_size,
                          context_size,
                          min_core_overlap,
                          offset)

    cores = builder.generate_cores()
    

    n = 0    
    for core in cores:
        vertices, edges = solver.get_subgraph(name_db,
                                              collection,
                                              x_lim=core.x_lim_context,
                                              y_lim=core.y_lim_context,
                                              z_lim=core.z_lim_context)

        g1, index_map = solver.subgraph_to_g1(vertices, 
                                              edges)
        g1_to_nml(g1, 
                  "./candidates2_core_{}.nml".format(n), 
                  knossos=True, 
                  voxel_size=[5.,5.,50.])
 
    
        solutions = solver.solve_subgraph(g1,
                                          index_map,
                                          distance_threshold=175,
                                          cc_min_vertices=4,
                                          start_edge_prior=160.0,
                                          selection_cost=-70.0,
                                          distance_factor=0.0,
                                          orientation_factor=15.0,
                                          comb_angle_factor=16.0,
                                          time_limit=300,
                                          hcs=False)

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

        vertices, edges = solver.get_subgraph(name_db,
                                              collection,
                                              x_lim={"min": offset[0], "max": 1024 * 5 + offset[0]},
                                              y_lim={"min": offset[1], "max": 1024 * 5 + offset[1]},
                                              z_lim={"min": offset[2], "max": 120 * 50 + offset[2]})

        g1, index_map = solver.subgraph_to_g1(vertices, edges, set_partner=False)
    
        g1_to_nml(g1, "./solution3_core{}.nml".format(n), knossos=True, voxel_size=[5.,5.,50.])
        n += 1


def clean_up(name_db, collection, x_lim, y_lim, z_lim):
    solver = CoreSolver()

    solver.remove_deg_0_vertices(name_db,
                                 collection,
                                 x_lim=x_lim,
                                 y_lim=y_lim,
                                 z_lim=z_lim)

    vertices, edges = solver.get_subgraph(name_db,
                                          collection,
                                          x_lim=x_lim,
                                          y_lim=y_lim,
                                          z_lim=z_lim)

    g1, index_map = solver.subgraph_to_g1(vertices, edges, set_partner=False)
    
    g1_to_nml(g1, "./solution_clean.nml", knossos=True, voxel_size=[5.,5.,50.])


def track(config_path):
    config = read_config(config_path)
    roi = [config["roi_x"], config["roi_y"], config["roi_z"]]
    
    if config["extract_candidates"]:
        """
        Check if candidate extraction needs to be performed
        otherwise it is assumed the given db collection holds
        already extracted candidates.
        """

        if config["prob_map_chunks_perp_dir"] is None:
            """
            Check if chunking of the probability maps needs to
            be performed, otherwise the chunk dir specified is used.
            """

            dir_perp, dir_par = chunk(volume_shape=config["volume_shape"],
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
        If all rois are [0,-1] extract candidates from
        all chunks
        """
        full_roi = np.array([False, False, False])
        for i in range(3):
            if list(roi[i]) == [0, -1]:
                full_roi[i] = True

        full_roi = np.all(full_roi)

        """
        Extract id, and volume information from chunks
        and compare with ROI
        """
        chunk_limits = {}
        chunk_ids = {}
        chunk_offsets = {}
        roi_offset = []
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

            # Determine roi offset relative to volume
            chunk_offset = [chunk_limits[j][0] for j in range(3)]
            if not roi_offset:
                roi_offset = chunk_offset
                
            for j in range(3):
                if chunk_offset[j] < roi_offset[j]:
                    roi_offset[j] = chunk_offset[j]

            chunk_limits[(f_chunk_perp, f_chunk_par)] = chunk_limit 
            chunk_ids[(f_chunk_perp, f_chunk_par)] = chunk_id
            chunk_offsets[]

            if full_roi:
                roi_chunks.append((f_chunk_perp, f_chunk_par))
            else:
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

    # Update roi volume size:
    roi_volume_size = [cl[1] - cl[0] for cl in chunk_limits.values()[0]] *\
                        config["voxel_size"] * len(roi_chunks)

    solve_candidate_volume(name_db=config["db_name"],
                           collection="candidates",
                           core_size=config["core_size"],
                           context_size=config["context_size"],
                           volume_size=roi_volume_size,
                           min_core_overlap=config["min_core_overlap"],
                           copy_candidates=True,
                           copy_target="microtubules",
                           offset=np.array())

    # Clean up all remaining degree 0 vertices found in the context area:
    clean_up(name_db=config["db_name"],
             collection="microtubules",
             x_lim=)  
         
        
        
        print chunk_ids
        print chunk_limits
        print roi_chunks 
            
        
 
if __name__ == "__main__":
    track("../config.ini")
