from mtrack.cores import DB, CoreBuilder
from mtrack.preprocessing import g1_to_nml
import os
import numpy as np

def retrieve(name_db,
             collection,
             x_lim,
             y_lim,
             z_lim,
             voxel_size,
             output_path):

    db = DB()
    g1_selected, index_map = db.get_selected(name_db,
                                             collection,
                                             x_lim=x_lim,
                                             y_lim=y_lim,
                                             z_lim=z_lim)

    print "Validate solution..."
    for v in g1_selected.get_vertex_iterator():
        assert(len(g1_selected.get_incident_edges(v)) <= 2), "Retrieved graph has branchings"
    print "...No violations"

    g1_to_nml(g1_selected, 
              output_path,
              knossos=True,
              voxel_size=voxel_size)

if __name__ == "__main__":
    retrieve("cremi_grid_search_run_rfc_0",
             "run_75",
             {"min": 0, "max": 1400 * 4},
             {"min": 0, "max": 1400 * 4},
             {"min": 0,"max": 130*40},
             [4.,4.,40.],
             "./rfc_75.nml")
