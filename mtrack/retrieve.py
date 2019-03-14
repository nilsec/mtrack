from mtrack.cores import DB, CoreBuilder
from mtrack.preprocessing import g1_to_nml
import os
import numpy as np
import pdb

def retrieve(name_db,
             collection,
             x_lim,
             y_lim,
             z_lim,
             voxel_size,
             output_path,
             selected_only=True):

    db = DB()
    if selected_only:
        g1, index_map = db.get_selected(name_db,
                                        collection,
                                        x_lim=x_lim,
                                        y_lim=y_lim,
                                        z_lim=z_lim)

        print "Validate solution..."
        for v in g1.get_vertex_iterator():
            assert(len(g1.get_incident_edges(v)) <= 2), "Retrieved graph has branchings"
        print "...No violations"

    else:
        g1, index_map = db.get_g1(name_db,
                                  collection,
                                  x_lim=x_lim,
                                  y_lim=y_lim,
                                  z_lim=z_lim)

        pdb.set_trace()



    g1_to_nml(g1, 
              output_path,
              knossos=True,
              voxel_size=voxel_size)

if __name__ == "__main__":
    retrieve("b+_full",
             "trial_0",
             {"min": 0, "max": 1200 * 4},
             {"min": 0, "max": 1200 * 4},
             {"min": 0,"max": 120*40},
             [4.,4.,40.],
             "./b+_candidates.nml",
             selected_only=False)
