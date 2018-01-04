import ConfigParser

config = ConfigParser.ConfigParser()
config.add_section('Data')
config.set('Data', 'prob_map_chunks_perp_dir', 'None')
config.set('Data', 'prob_map_chunks_par_dir', 'None')
config.set('Data', 'prob_maps_perp_dir', './prob_maps/perp')
config.set('Data', 'prob_maps_par_dir', './prob_maps/par')
config.set('Data', 'db_name', 'volume')
config.set('Data', 'overwrite_candidates', 'False')
config.set('Data', 'extract_candidates', 'True')

config.add_section('Preprocessing')
config.set('Preprocessing', 'gaussian_sigma_perp', '0.5')
config.set('Preprocessing', 'gaussian_sigma_par', '0.5')
config.set('Preprocessing', 'point_threshold_perp', '0.45')
config.set('Preprocessing', 'point_threshold_par', '0.45')
config.set('Preprocessing', 'distance_threshold', '175')

config.add_section('Chunks')
config.set('Chunks', 'volume_shape', '0, 0, 0')
config.set('Chunks', 'max_chunk_shape', '0, 0, 0')
config.set('Chunks', 'chunk_overlap', '0, 0, 0')
config.set('Chunks', 'chunk_output_dir', './chunks')

config.add_section('Cores')
config.set('Cores', 'core_size', '400, 400, 40')
config.set('Cores', 'context_size', '40, 40, 4')
config.set('Cores', 'min_core_overlap', '0, 0, 0')
config.set('Cores', 'voxel_size', '5., 5., 50.')

config.add_section('Output')
config.set('Output', 'roi_x', '0, -1')
config.set('Output', 'roi_y', '0, -1')
config.set('Output', 'roi_z', '0, -1')
config.set('Output', 'output_dir', './solution')
config.set('Output', 'nml', 'True')
config.set('Output', 'gt', 'True')

config.add_section('Solve')
config.set('Solve', 'cc_min_vertices', '4')
config.set('Solve', 'start_edge_prior', '160.0')
config.set('Solve', 'selection_cost', '-70.0')
config.set('Solve', 'distance_factor', '0.0')
config.set('Solve', 'orientation_factor', '15.0')
config.set('Solve', 'comb_angle_factor', '16.0')
config.set('Solve', 'time_limit_per_cc', '1000')
config.set('Solve', 'get_hcs', 'False')

with open('../../config.ini', 'wb') as configfile:
    config.write(configfile)

