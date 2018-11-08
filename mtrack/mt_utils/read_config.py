import ConfigParser
import numpy as np

def read_config(path):
    with open(path) as fp:
        config = ConfigParser.ConfigParser()
        config.readfp(fp)

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
        cfg_dict["eval_output_dir"] = config.get("Evaluate", "eval_output_dir")
        

        # Ilastik
        cfg_dict["extract_perp"] = config.getboolean("Ilastik", "extract_perp")
        cfg_dict["extract_par"] = config.getboolean("Ilastik", "extract_par")
        cfg_dict["pm_output_dir_perp"] = config.get("Ilastik", "pm_output_dir_perp")
        cfg_dict["pm_output_dir_par"] = config.get("Ilastik", "pm_output_dir_par")
        cfg_dict["ilastik_source_dir"] = config.get("Ilastik", "ilastik_source_dir")
        cfg_dict["ilastik_project_perp"] = config.get("Ilastik", "ilastik_project_perp")
        cfg_dict["ilastik_project_par"] = config.get("Ilastik", "ilastik_project_par")
        cfg_dict["raw"] = config.get("Ilastik", "raw")
        cfg_dict["file_extension"] = config.get("Ilastik", "file_extension")
        cfg_dict["h5_dset"] = config.get("Ilastik", "h5_dset")
        cfg_dict["label"] = config.getint("Ilastik", "label")

        # Data
        cfg_dict["prob_map_chunks_perp_dir"] = config.get("Data", "prob_map_chunks_perp_dir")
        cfg_dict["prob_map_chunks_par_dir"] = config.get("Data", "prob_map_chunks_par_dir")
        cfg_dict["perp_stack_h5"] = config.get("Data", "perp_stack_h5")
        cfg_dict["par_stack_h5"] = config.get("Data", "par_stack_h5")
        cfg_dict["name_db"] = config.get("Data", "name_db")
        cfg_dict["name_collection"] = config.get("Data", "name_collection")
        cfg_dict["extract_candidates"] = config.getboolean("Data", "extract_candidates")
        cfg_dict["reset"] = config.getboolean("Data", "reset")

        # Preprocessing
        cfg_dict["gaussian_sigma_perp"] = config.getfloat("Preprocessing", "gaussian_sigma_perp")
        cfg_dict["gaussian_sigma_par"] = config.getfloat("Preprocessing", "gaussian_sigma_par")
        cfg_dict["point_threshold_perp"] = config.getfloat("Preprocessing", "point_threshold_perp")
        cfg_dict["point_threshold_par"] = config.getfloat("Preprocessing", "point_threshold_par")
        cfg_dict["distance_threshold"] = config.getint("Preprocessing", "distance_threshold")
        
        # Chunks
        cfg_dict["volume_shape"] = np.array(config.get("Chunks", "volume_shape").split(", "), dtype=int)
        cfg_dict["max_chunk_shape"] = np.array(config.get("Chunks", "max_chunk_shape").split(", "), dtype=int)
        cfg_dict["chunk_output_dir"] = config.get("Chunks", "chunk_output_dir")

        # Cores
        cfg_dict["core_size"] = np.array(config.get("Cores", "core_size").split(", "), dtype=int)
        cfg_dict["context_size"] = np.array(config.get("Cores", "context_size").split(", "), dtype=int)
        cfg_dict["voxel_size"] = np.array(config.get("Cores", "voxel_size").split(", "), dtype=float)
        
        # Output
        cfg_dict["roi_x"] = np.array(config.get("Output", "roi_x").split(", "), dtype=int)
        cfg_dict["roi_y"] = np.array(config.get("Output", "roi_y").split(", "), dtype=int)
        cfg_dict["roi_z"] = np.array(config.get("Output", "roi_z").split(", "), dtype=int)
        
        # Solve
        cfg_dict["solve"] = config.getboolean("Solve", "solve")
        cfg_dict["backend"] = config.get("Solve", "backend")
        cfg_dict["mp"] = config.getboolean("Solve", "mp")
        cfg_dict["validate_selection"] = config.getboolean("Solve", "validate_selection")
        cfg_dict["export_validated"] = config.getboolean("Solve", "export_validated")
        cfg_dict["validated_output_path"] = config.get("Solve", "validated_output_path")
        cfg_dict["cc_min_vertices"] = config.getint("Solve", "cc_min_vertices")
        cfg_dict["start_edge_prior"] = config.getfloat("Solve", "start_edge_prior")
        cfg_dict["selection_cost"] = config.getfloat("Solve", "selection_cost")
        cfg_dict["distance_factor"] = config.getfloat("Solve", "distance_factor")
        cfg_dict["orientation_factor"] = config.getfloat("Solve", "orientation_factor")
        cfg_dict["comb_angle_factor"] = config.getfloat("Solve", "comb_angle_factor")
        cfg_dict["time_limit_per_cc"] = config.getint("Solve", "time_limit_per_cc")

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
    read_config("../../config.ini")   
