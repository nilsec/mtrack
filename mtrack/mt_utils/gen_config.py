import ConfigParser
import os

def gen_config(prob_map,
               prob_map_dset,
               name_db,
               name_collection,
               extract_candidates,
               reset,
               db_credentials,
               distance_threshold,
               threshold,
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
    config.set('Data', 'prob_map', str(prob_map))
    config.set('Data', 'prob_map_dset', str(prob_map_dset))
    config.set('Data', 'name_db', str(name_db))
    config.set('Data', 'name_collection', str(name_collection))
    config.set('Data', 'extract_candidates', str(extract_candidates))
    config.set('Data', 'reset', str(reset))
    config.set('Data', 'db_credentials', str(db_credentials))

    config.add_section('Preprocessing')
    config.set('Preprocessing', 'distance_threshold', str(distance_threshold))
    config.set('Preprocessing', 'threshold', str(threshold))

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
    gen_config(prob_map=None,
               prob_map_dset=None,
               name_db="db_name",
               name_collection="collection_name",
               extract_candidates=True,
               reset=False,
               db_credentials="path_to_db_credentials.ini",
               distance_threshold=175,
               threshold=0.5,
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
