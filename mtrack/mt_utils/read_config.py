import ConfigParser
import numpy as np

def read_config(path):
    config = ConfigParser.ConfigParser()
    config.read(path)

    cfg_dict = {}

    # Data
    cfg_dict["prob_map_chunks_perp_dir"] = config.get("Data", "prob_map_chunks_perp_dir")
    cfg_dict["prob_map_chunks_par_dir"] = config.get("Data", "prob_map_chunks_par_dir")
    cfg_dict["prob_maps_perp_dir"] = config.get("Data", "prob_maps_perp_dir")
    cfg_dict["prob_maps_par_dir"] = config.get("Data", "prob_maps_par_dir")
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
    cfg_dict["save_roi"] = config.getboolean("Output", "save_roi")
    cfg_dict["nml"] = config.getboolean("Output", "nml")
    cfg_dict["gt"] = config.getboolean("Output", "gt")

    # Solve
    cfg_dict["cc_min_vertices"] = config.getint("Solve", "cc_min_vertices")
    cfg_dict["start_edge_prior"] = config.getfloat("Solve", "start_edge_prior")
    cfg_dict["selection_cost"] = config.getfloat("Solve", "selection_cost")
    cfg_dict["distance_factor"] = config.getfloat("Solve", "distance_factor")
    cfg_dict["orientation_factor"] = config.getfloat("Solve", "orientation_factor")
    cfg_dict["comb_angle_factor"] = config.getfloat("Solve", "comb_angle_factor")
    cfg_dict["time_limit_per_cc"] = config.getint("Solve", "time_limit_per_cc")
    cfg_dict["get_hcs"] = config.getboolean("Solve", "get_hcs")

    return cfg_dict

if __name__ == "__main__":
    read_config_2("../../config.ini")   
