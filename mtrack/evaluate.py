from mtrack.cores import DB
from mtrack.evaluation import evaluate
from mtrack.preprocessing import nml_to_g1, g1_to_nml

import numpy as np


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
                 output_dir,
                 solution_file=None):

    if solution_file is None:
        db = DB()

        g1, index_map = db.get_selected(name_db,
                                    collection,
                                    x_lim=x_lim,
                                    y_lim=y_lim,
                                    z_lim=z_lim)
        scale_solution = True
    else:
        g1 = solution_file
        scale_solution = False

    tracing_cut = cut_to_roi(tracing_file, x_lim, y_lim, z_lim, voxel_size, output_dir)

    evaluate(tracing_file=tracing_cut,
             solution_file=g1,
             scale_solution_to_voxel=scale_solution,
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


def cut_to_roi(tracing, x_lim, y_lim, z_lim, voxel_size, output_dir):
    g1_tracing = nml_to_g1(tracing, None) 

    min_roi = np.array([x_lim["min"], y_lim["min"], z_lim["min"]])
    max_roi = np.array([x_lim["max"], y_lim["max"], z_lim["max"]])
    min_roi = np.array([min_roi[i]/voxel_size[i] for i in range(3)])
    max_roi = np.array([max_roi[i]/voxel_size[i] for i in range(3)])

    in_roi_vp = g1_tracing.new_vertex_property("in_roi", "bool", False)
    for v in g1_tracing.get_vertex_iterator():
        position = g1_tracing.get_position(v)
        if np.all(position>=min_roi) and np.all(position<=max_roi):
            in_roi_vp[v] = True
        else:
            in_roi_vp[v] = False

    g1_tracing.set_vertex_filter(in_roi_vp)
    g1_to_nml(g1_tracing, output_dir + "/tracing_eval_crop.nml", knossify=True)
    return g1_tracing
