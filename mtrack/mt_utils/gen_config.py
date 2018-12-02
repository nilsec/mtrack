import ConfigParser
import os

def gen_config(candidate_extraction_mode,
               prob_map_chunks_single_dir,
               prob_map_chunks_perp_dir,
               prob_map_chunks_par_dir,
               single_stack_h5,
               perp_stack_h5,
               par_stack_h5,
               name_db,
               name_collection,
               extract_candidates,
               reset,
               gaussian_sigma_single,
               gaussian_sigma_perp,
               gaussian_sigma_par,
               point_threshold_single,
               point_threshold_perp,
               point_threshold_par,
               distance_threshold,
               volume_shape,
               volume_offset,
               max_chunk_shape,
               chunk_output_dir,
               core_size,
               context_size,
               voxel_size,
               roi_x,
               roi_y,
               roi_z,
               solve,
               validate_selection,
               export_validated,
               validated_output_path,
               backend,
               mp,  
               cc_min_vertices,
               start_edge_prior,
               selection_cost,
               distance_factor,
               orientation_factor,
               comb_angle_factor,
               time_limit_per_cc,
               cfg_output_dir):

    config = ConfigParser.ConfigParser()

    config.add_section('Data')
    config.set('Data', 'single_stack_h5', str(single_stack_h5))
    config.set('Data', 'perp_stack_h5', str(perp_stack_h5))
    config.set('Data', 'par_stack_h5', str(par_stack_h5))
    config.set('Data', 'candidate_extraction_mode', str(candidate_extraction_mode))
    config.set('Data', 'name_db', str(name_db))
    config.set('Data', 'name_collection', str(name_collection))
    config.set('Data', 'extract_candidates', str(extract_candidates))
    config.set('Data', 'reset', str(reset))

    config.add_section('Preprocessing')
    config.set('Preprocessing', 'gaussian_sigma_single', str(gaussian_sigma_single))
    config.set('Preprocessing', 'gaussian_sigma_perp', str(gaussian_sigma_perp))
    config.set('Preprocessing', 'gaussian_sigma_par', str(gaussian_sigma_par))
    config.set('Preprocessing', 'point_threshold_single', str(point_threshold_single))
    config.set('Preprocessing', 'point_threshold_perp', str(point_threshold_perp))
    config.set('Preprocessing', 'point_threshold_par', str(point_threshold_par))
    config.set('Preprocessing', 'distance_threshold', str(distance_threshold))

    config.add_section('Chunks')
    config.set('Chunks', 'volume_shape', str(volume_shape[0]) + ", " +\
                                         str(volume_shape[1]) + ", " +\
                                         str(volume_shape[2]))
    config.set('Chunks', 'volume_offset', str(volume_offset[0]) + ", " +\
                                          str(volume_offset[1]) + ", " +\
                                          str(volume_offset[2]))
    config.set('Chunks', 'max_chunk_shape', str(max_chunk_shape[0]) + ", " +\
                                            str(max_chunk_shape[1]) + ", " +\
                                            str(max_chunk_shape[2]))

    config.set('Chunks', 'chunk_output_dir', str(chunk_output_dir))
    config.set('Chunks', 'prob_map_chunks_single_dir', str(prob_map_chunks_single_dir))
    config.set('Chunks', 'prob_map_chunks_perp_dir', str(prob_map_chunks_perp_dir))
    config.set('Chunks', 'prob_map_chunks_par_dir', str(prob_map_chunks_par_dir))

    config.add_section('Cores')
    config.set('Cores', 'core_size', str(core_size[0]) + ", " +\
                                     str(core_size[1]) + ", " +\
                                     str(core_size[2]))
    config.set('Cores', 'context_size', str(context_size[0]) + ", " +\
                                        str(context_size[1]) + ", " +\
                                        str(context_size[2]))
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
    config.set('Solve', 'validate_selection', str(validate_selection))
    config.set('Solve', 'export_validated', str(export_validated))
    config.set('Solve', 'validated_output_path', str(validated_output_path))
    config.set('Solve', 'cc_min_vertices', str(cc_min_vertices))
    config.set('Solve', 'start_edge_prior', str(start_edge_prior))
    config.set('Solve', 'selection_cost', str(selection_cost))
    config.set('Solve', 'distance_factor', str(distance_factor))
    config.set('Solve', 'orientation_factor', str(orientation_factor))
    config.set('Solve', 'comb_angle_factor', str(comb_angle_factor))
    config.set('Solve', 'time_limit_per_cc', str(time_limit_per_cc))

    if not os.path.exists(cfg_output_dir):
        os.makedirs(cfg_output_dir)

    with open(str(cfg_output_dir) + '/config.ini', 'wb') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    gen_config(candidate_extraction_mode="single/double",
               prob_map_chunks_single_dir=None,
               prob_map_chunks_perp_dir=None,
               prob_map_chunks_par_dir=None,
               single_stack_h5=None,
               perp_stack_h5=None,
               par_stack_h5=None,
               name_db="db_name",
               name_collection="collection_name",
               extract_candidates=True,
               reset=False,
               gaussian_sigma_single=0.5,
               gaussian_sigma_perp=0.5,
               gaussian_sigma_par=0.5,
               point_threshold_single=0.5,
               point_threshold_perp=0.6,
               point_threshold_par=0.6,
               distance_threshold=175,
               volume_shape=[154,1524,1524],
               volume_offset=[0,0,0],
               max_chunk_shape=[50,1524,1524],
               chunk_output_dir="path_to_dir",
               core_size=[300,300,30],
               context_size=[50,50,5],
               voxel_size=[5.,5.,50.],
               roi_x=[0,700],
               roi_y=[0,700],
               roi_z=[0,40],
               solve=True,
               backend="Gurobi",
               mp=True,  
               validate_selection="True",
               export_validated=False,
               validated_output_path="path_to_output_file",
               cc_min_vertices=4,
               start_edge_prior=160.0,
               selection_cost=-70.0,
               distance_factor=0.0,
               orientation_factor=15.0,
               comb_angle_factor=16.0,
               time_limit_per_cc=1000,
               cfg_output_dir="../../")
