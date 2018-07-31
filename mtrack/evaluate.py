from mtrack.cores import DB
from mtrack.evaluation import evaluate

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

    db = DB()

    g1, index_map = db.get_selected(name_db,
                                    collection,
                                    x_lim=x_lim,
                                    y_lim=y_lim,
                                    z_lim=z_lim)

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
