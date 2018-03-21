import ConfigParser
import numpy as np

def read_config(path):
    config = ConfigParser.ConfigParser()
    config.read(path)

    cfg_dict = {}

    # Evaluate
    cfg_dict["evaluate"] = config.getboolean("Evaluate", "evaluate")
    cfg_dict["tracing_file"] = config.get("Evaluate", "tracing_file")
    cfg_dict["eval_chunk_size"] = config.getint("Evaluate", "eval_chunk_size")
    cfg_dict["eval_distance_tolerance"] = config.getfloat("Evaluate", "eval_distance_tolerance")
    cfg_dict["eval_dummy_cost"] = config.getint("Evaluate", "eval_dummy_cost")
    cfg_dict["eval_edge_selection_cost"] = config.getfloat("Evaluate", "eval_edge_selection_cost")
    cfg_dict["eval_pair_cost_factor"] = config.getfloat("Evaluate", "eval_pair_cost_factor")
    cfg_dict["max_edges"] = config.getint("Evaluate", "max_edges")
    cfg_dict["eval_time_limit"] = config.getint("Evaluate", "eval_time_limit") 
    

    # Ilastik
    cfg_dict["extract_perp"] = config.getboolean("Ilastik", "extract_perp")
    cfg_dict["extract_par"] = config.getboolean("Ilastik", "extract_par")
    cfg_dict["pm_output_dir_perp"] = config.get("Ilastik", "pm_output_dir_perp")
    cfg_dict["pm_output_dir_par"] = config.get("Ilastik", "pm_output_dir_par")
    cfg_dict["ilastik_source_dir"] = config.get("Ilastik", "ilastik_source_dir")
    cfg_dict["ilastik_project_perp"] = config.get("Ilastik", "ilastik_project_perp")
    cfg_dict["ilastik_project_par"] = config.get("Ilastik", "ilastik_project_par")
    cfg_dict["image_dir"] = config.get("Ilastik", "image_dir")
    cfg_dict["file_extension"] = config.get("Ilastik", "file_extension")
    cfg_dict["h5_input_path"] = config.get("Ilastik", "h5_input_path")

    # Data
    cfg_dict["prob_map_chunks_perp_dir"] = config.get("Data", "prob_map_chunks_perp_dir")
    cfg_dict["prob_map_chunks_par_dir"] = config.get("Data", "prob_map_chunks_par_dir")
    cfg_dict["perp_stack_h5"] = config.get("Data", "perp_stack_h5")
    cfg_dict["par_stack_h5"] = config.get("Data", "par_stack_h5")
    cfg_dict["db_name"] = config.get("Data", "db_name")
    cfg_dict["overwrite_candidates"] = config.getboolean("Data", "overwrite_candidates")
    cfg_dict["extract_candidates"] = config.getboolean("Data", "extract_candidates")
    cfg_dict["overwrite_copy_target"] = config.getboolean("Data", "overwrite_copy_target")
    cfg_dict["skip_solved_cores"] = config.getboolean("Data", "skip_solved_cores")

    # Preprocessing
    cfg_dict["gaussian_sigma_perp"] = config.getfloat("Preprocessing", "gaussian_sigma_perp")
    cfg_dict["gaussian_sigma_par"] = config.getfloat("Preprocessing", "gaussian_sigma_par")
    cfg_dict["point_threshold_perp"] = config.getfloat("Preprocessing", "point_threshold_perp")
    cfg_dict["point_threshold_par"] = config.getfloat("Preprocessing", "point_threshold_par")
    cfg_dict["distance_threshold"] = config.getint("Preprocessing", "distance_threshold")
    
    # Chunks
    cfg_dict["volume_shape"] = np.array(config.get("Chunks", "volume_shape").split(", "), dtype=int)
    cfg_dict["max_chunk_shape"] = np.array(config.get("Chunks", "max_chunk_shape").split(", "), dtype=int)
    cfg_dict["chunk_overlap"] = np.array(config.get("Chunks", "chunk_overlap").split(", "), dtype=int)
    cfg_dict["chunk_output_dir"] = config.get("Chunks", "chunk_output_dir")

    # Cores
    cfg_dict["core_wise"] = config.getboolean("Cores", "core_wise")
    cfg_dict["core_size"] = np.array(config.get("Cores", "core_size").split(", "), dtype=int)
    cfg_dict["context_size"] = np.array(config.get("Cores", "context_size").split(", "), dtype=int)
    cfg_dict["min_core_overlap"] = np.array(config.get("Cores", "min_core_overlap").split(", "), dtype=int)
    cfg_dict["voxel_size"] = np.array(config.get("Cores", "voxel_size").split(", "), dtype=float)
    
    # Output
    cfg_dict["roi_x"] = np.array(config.get("Output", "roi_x").split(", "), dtype=int)
    cfg_dict["roi_y"] = np.array(config.get("Output", "roi_y").split(", "), dtype=int)
    cfg_dict["roi_z"] = np.array(config.get("Output", "roi_z").split(", "), dtype=int)
    cfg_dict["output_dir"] = config.get("Output", "output_dir")
    cfg_dict["save_cores"] = config.getboolean("Output", "save_cores")
    cfg_dict["save_candidates"] = config.getboolean("Output", "save_candidates")
    cfg_dict["save_connected"] = config.getboolean("Output", "save_connected")
    cfg_dict["save_core_graphs"] = config.getboolean("Output", "save_core_graphs")
    cfg_dict["save_roi"] = config.getboolean("Output", "save_roi")
    cfg_dict["nml"] = config.getboolean("Output", "nml")
    cfg_dict["gt"] = config.getboolean("Output", "gt")
    
    # Solve
    cfg_dict["solve"] = config.getboolean("Solve", "solve")
    cfg_dict["mp"] = config.getboolean("Solve", "mp")
    cfg_dict["cc_min_vertices"] = config.getint("Solve", "cc_min_vertices")
    cfg_dict["start_edge_prior"] = config.getfloat("Solve", "start_edge_prior")
    cfg_dict["selection_cost"] = config.getfloat("Solve", "selection_cost")
    cfg_dict["distance_factor"] = config.getfloat("Solve", "distance_factor")
    cfg_dict["orientation_factor"] = config.getfloat("Solve", "orientation_factor")
    cfg_dict["comb_angle_factor"] = config.getfloat("Solve", "comb_angle_factor")
    cfg_dict["time_limit_per_cc"] = config.getint("Solve", "time_limit_per_cc")
    cfg_dict["get_hcs"] = config.getboolean("Solve", "get_hcs")

    # Cluster
    cfg_dict["cluster"] = config.getboolean("Cluster", "cluster")
    cfg_dict["cluster_output_dir"] = config.get("Cluster", "cluster_output_dir")
    cfg_dict["epsilon_lines"] = config.getint("Cluster", "epsilon_lines")
    cfg_dict["epsilon_volumes"] = config.getint("Cluster", "epsilon_volumes")
    cfg_dict["min_overlap_volumes"] = config.getint("Cluster", "min_overlap_volumes")
    cfg_dict["cluster_orientation_factor"] = config.getint("Cluster", "cluster_orientation_factor")
    cfg_dict["remove_singletons"] = config.getint("Cluster", "remove_singletons")
    cfg_dict["use_ori"] = config.getboolean("Cluster", "use_ori")
    

    return cfg_dict

if __name__ == "__main__":
    read_config_2("../../config.ini")   
