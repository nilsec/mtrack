import ConfigParser
import os

def gen_config(evaluate,
               tracing_file,
               eval_chunk_size,
               eval_distance_tolerance,
               eval_dummy_cost,
               eval_edge_selection_cost,
               eval_pair_cost_factor,
               max_edges,
               eval_time_limit,
               eval_output_dir,
               extract_perp,
               extract_par,
               pm_output_dir_perp,
               pm_output_dir_par,
               ilastik_source_dir,
               ilastik_project_perp,
               ilastik_project_par,
               raw,
               file_extension,
               h5_dset,
               label,
               prob_map_chunks_perp_dir,
               prob_map_chunks_par_dir,
               perp_stack_h5,
               par_stack_h5,
               db_name,
               overwrite_candidates,
               extract_candidates,
               overwrite_copy_target,
               skip_solved_cores,
               reset,
               gaussian_sigma_perp,
               gaussian_sigma_par,
               point_threshold_perp,
               point_threshold_par,
               distance_threshold,
               volume_shape,
               max_chunk_shape,
               chunk_output_dir,
               core_size,
               context_size,
               voxel_size,
               roi_x,
               roi_y,
               roi_z,
               solve,
               backend,
               mp,  
               cc_min_vertices,
               start_edge_prior,
               selection_cost,
               distance_factor,
               orientation_factor,
               comb_angle_factor,
               time_limit_per_cc,
               get_hcs,
               cluster,
               cluster_output_dir,
               epsilon_lines,
               epsilon_volumes,
               min_overlap_volumes,
               cluster_orientation_factor,
               remove_singletons,
               use_ori,
               cfg_output_dir):

    config = ConfigParser.ConfigParser()

    config.add_section('Evaluate')
    config.set('Evaluate', 'evaluate', str(evaluate))
    config.set('Evaluate', 'tracing_file', str(tracing_file))
    config.set('Evaluate', 'eval_chunk_size', str(eval_chunk_size))
    config.set('Evaluate', 'eval_distance_tolerance', str(eval_distance_tolerance))
    config.set('Evaluate', 'eval_dummy_cost', str(eval_dummy_cost))
    config.set('Evaluate', 'eval_edge_selection_cost', str(eval_edge_selection_cost))
    config.set('Evaluate', 'eval_pair_cost_factor', str(eval_pair_cost_factor))
    config.set('Evaluate', 'max_edges', str(max_edges))
    config.set('Evaluate', 'eval_time_limit', str(eval_time_limit))
    config.set('Evaluate', 'eval_output_dir', str(eval_output_dir))

    config.add_section('Ilastik')
    config.set('Ilastik', 'extract_perp', str(extract_perp))
    config.set('Ilastik', 'extract_par', str(extract_par))
    config.set('Ilastik', 'pm_output_dir_perp', str(pm_output_dir_perp))
    config.set('Ilastik', 'pm_output_dir_par', str(pm_output_dir_par))
    config.set('Ilastik', 'ilastik_source_dir', str(ilastik_source_dir))
    config.set('Ilastik', 'ilastik_project_perp', str(ilastik_project_perp))
    config.set('Ilastik', 'ilastik_project_par', str(ilastik_project_par))
    config.set('Ilastik', 'raw', str(raw))
    config.set('Ilastik', 'file_extension', str(file_extension))
    config.set('Ilastik', 'h5_dset', str(h5_dset))
    config.set('Ilastik', 'label', str(label))
 

    config.add_section('Data')
    config.set('Data', 'prob_map_chunks_perp_dir', str(prob_map_chunks_perp_dir))
    config.set('Data', 'prob_map_chunks_par_dir', str(prob_map_chunks_par_dir))
    config.set('Data', 'perp_stack_h5', str(perp_stack_h5))
    config.set('Data', 'par_stack_h5', str(par_stack_h5))
    config.set('Data', 'db_name', str(db_name))
    config.set('Data', 'overwrite_candidates', str(overwrite_candidates))
    config.set('Data', 'extract_candidates', str(extract_candidates))
    config.set('Data', 'overwrite_copy_target', str(overwrite_copy_target))
    config.set('Data', 'skip_solved_cores', str(skip_solved_cores))
    config.set('Data', 'reset', str(reset))

    config.add_section('Preprocessing')
    config.set('Preprocessing', 'gaussian_sigma_perp', str(gaussian_sigma_perp))
    config.set('Preprocessing', 'gaussian_sigma_par', str(gaussian_sigma_par))
    config.set('Preprocessing', 'point_threshold_perp', str(point_threshold_perp))
    config.set('Preprocessing', 'point_threshold_par', str(point_threshold_par))
    config.set('Preprocessing', 'distance_threshold', str(distance_threshold))

    config.add_section('Chunks')
    config.set('Chunks', 'volume_shape', str(volume_shape[0]) + ", " +\
                                         str(volume_shape[1]) + ", " +\
                                         str(volume_shape[2]))
    config.set('Chunks', 'max_chunk_shape', str(max_chunk_shape[0]) + ", " +\
                                            str(max_chunk_shape[1]) + ", " +\
                                            str(max_chunk_shape[2]))

    config.set('Chunks', 'chunk_overlap', '0, 0, 0')
    config.set('Chunks', 'chunk_output_dir', str(chunk_output_dir))

    config.add_section('Cores')
    config.set('Cores', 'core_size', str(core_size[0]) + ", " +\
                                     str(core_size[1]) + ", " +\
                                     str(core_size[2]))
    config.set('Cores', 'context_size', str(context_size[0]) + ", " +\
                                        str(context_size[1]) + ", " +\
                                        str(context_size[2]))
    config.set('Cores', 'min_core_overlap', '0, 0, 0')
    config.set('Cores', 'voxel_size', str(voxel_size[0]) + ", " +\
                                      str(voxel_size[1]) + ", " +\
                                      str(voxel_size[2]))

    config.add_section('Output')
    config.set('Output', 'roi_x', str(roi_x[0]) + ', ' + str(roi_x[1]))
    config.set('Output', 'roi_y', str(roi_y[0]) + ', ' + str(roi_y[1]))
    config.set('Output', 'roi_z', str(roi_z[0]) + ', ' + str(roi_z[1]))

    config.add_section('Solve')
    config.set('Solve', 'solve', str(solve))
    config.set('Solve', 'backend', str(backend))
    config.set('Solve', 'mp', str(mp))
    config.set('Solve', 'cc_min_vertices', str(cc_min_vertices))
    config.set('Solve', 'start_edge_prior', str(start_edge_prior))
    config.set('Solve', 'selection_cost', str(selection_cost))
    config.set('Solve', 'distance_factor', str(distance_factor))
    config.set('Solve', 'orientation_factor', str(orientation_factor))
    config.set('Solve', 'comb_angle_factor', str(comb_angle_factor))
    config.set('Solve', 'time_limit_per_cc', str(time_limit_per_cc))
    config.set('Solve', 'get_hcs', str(get_hcs))

    config.add_section('Cluster')
    config.set('Cluster', 'cluster', str(cluster))
    config.set('Cluster', 'cluster_output_dir', str(cluster_output_dir))
    config.set('Cluster', 'epsilon_lines', str(epsilon_lines))
    config.set('Cluster', 'epsilon_volumes', str(epsilon_volumes))
    config.set('Cluster', 'min_overlap_volumes', str(min_overlap_volumes))
    config.set('Cluster', 'cluster_orientation_factor', str(cluster_orientation_factor))
    config.set('Cluster', 'remove_singletons', str(remove_singletons))
    config.set('Cluster', 'use_ori', str(use_ori))

    if not os.path.exists(cfg_output_dir):
        os.makedirs(cfg_output_dir)

    with open(str(cfg_output_dir) + '/config.ini', 'wb') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    gen_config(evaluate=False,
               tracing_file=None,
               eval_chunk_size=10,
               eval_distance_tolerance=100,
               eval_dummy_cost=10000000,
               eval_edge_selection_cost=-10.0,
               eval_pair_cost_factor=1.0,
               max_edges=3,
               eval_time_limit=5000,
               eval_output_dir=None,
               extract_perp=False,
               extract_par=False,
               pm_output_dir_perp=None,
               pm_output_dir_par=None,
               ilastik_source_dir=None,
               ilastik_project_perp=None,
               ilastik_project_par=None,
               raw=None,
               file_extension=None,
               h5_dset=None,
               label=0,
               prob_map_chunks_perp_dir="/media/nilsec/d0/gt_mt_data/probability_maps/validation/perpendicular/stack",
               prob_map_chunks_par_dir="/media/nilsec/d0/gt_mt_data/probability_maps/validation/parallel/stack",
               perp_stack_h5="stack_corrected.h5",
               par_stack_h5="stack_corrected.h5",
               db_name="l3_val_test",
               overwrite_candidates=True,
               extract_candidates=True,
               overwrite_copy_target=True,
               skip_solved_cores=False,
               reset=False,
               gaussian_sigma_perp=0.5,
               gaussian_sigma_par=0.5,
               point_threshold_perp=0.6,
               point_threshold_par=0.6,
               distance_threshold=175,
               volume_shape=[154,1524,1524],
               max_chunk_shape=[50,1524,1524],
               chunk_output_dir="/media/nilsec/d0/gt_mt_data/mtrack/run_0/chunks",
               core_size=[300,300,30],
               context_size=[50,50,5],
               voxel_size=[5.,5.,50.],
               roi_x=[0,700],
               roi_y=[0,700],
               roi_z=[0,40],
               solve=True,
               backend="Scip",
               mp=True,  
               cc_min_vertices=4,
               start_edge_prior=160.0,
               selection_cost=-70.0,
               distance_factor=0.0,
               orientation_factor=15.0,
               comb_angle_factor=16.0,
               time_limit_per_cc=1000,
               get_hcs=False,
               cluster=False,
               cluster_output_dir=None,
               epsilon_lines=150,
               epsilon_volumes=100,
               min_overlap_volumes=1,
               cluster_orientation_factor=1000,
               remove_singletons=1,
               use_ori=True,
               cfg_output_dir="../../")
