from mtrack.cores import DB, CoreBuilder
from mtrack.preprocessing import g1_to_nml
import os
import numpy as np

def retrieve(name_db,
             collection,
             db_credentials,
             x_lim,
             y_lim,
             z_lim,
             voxel_size,
             output_path,
             selected_only=True):

    db = DB(db_credentials)
    if selected_only:
        g1, index_map = db.get_selected(name_db,
                                        collection,
                                        x_lim=x_lim,
                                        y_lim=y_lim,
                                        z_lim=z_lim)

        for v in g1.get_vertex_iterator():
            assert(len(g1.get_incident_edges(v)) <= 2), "Retrieved graph has branchings"

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
