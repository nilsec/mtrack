import ConfigParser
import numpy as np

def read_config(path):
    with open(path) as fp:
        config = ConfigParser.ConfigParser()
        config.readfp(fp)

        cfg_dict = {}

        # Data
        cfg_dict["maxima"] = config.get("Data", "maxima")
        cfg_dict["maxima_dset"] = config.get("Data", "maxima_dset")
        cfg_dict["prob_map"] = config.get("Data", "prob_map")
        cfg_dict["prob_map_dset"] = config.get("Data", "prob_map_dset")
        cfg_dict["name_db"] = config.get("Data", "name_db")
        cfg_dict["name_collection"] = config.get("Data", "name_collection")
        cfg_dict["extract_candidates"] = config.getboolean("Data", "extract_candidates")
        cfg_dict["reset"] = config.getboolean("Data", "reset")
        cfg_dict["db_credentials"] = config.get("Data", "db_credentials")

        # Preprocessing
        cfg_dict["distance_threshold"] = config.getint("Preprocessing", "distance_threshold")
        
        # Chunks
        cfg_dict["volume_shape"] = np.array(config.get("Chunks", "volume_shape").split(", "), dtype=int)
        cfg_dict["volume_offset"] = np.array(config.get("Chunks", "volume_offset").split(", "), dtype=int)
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

    return cfg_dict

if __name__ == "__main__":
    print read_config("../../config.ini")   
