from mtrack.cores import CoreSolver
from mtrack.preprocessing import g1_to_nml, connect_graph_locally
import os

def retrieve(name_db,
             collection,
             x_lim,
             y_lim,
             z_lim,
             voxel_size,
             save_dir,
             distance_threshold=None):

    solver = CoreSolver()
    vertices, edges = solver.get_subgraph(name_db,
                                          collection,
                                          x_lim=x_lim,
                                          y_lim=y_lim,
                                          z_lim=z_lim)

    g1, index_map = solver.subgraph_to_g1(vertices, edges)

    g1_to_nml(g1, 
              os.path.join(save_dir, "{}_{}_small.nml".format(name_db, collection)),
              knossos=True,
              voxel_size=voxel_size)

    if distance_threshold is not None:
        g1 = connect_graph_locally(g1, distance_threshold)
        g1_to_nml(g1,
                  os.path.join(save_dir, "{}_{}_dt{}.nml".format(name_db, 
                                                                 collection, 
                                                                 distance_threshold)),
                  knossos=True,
                  voxel_size=voxel_size)

if __name__ == "__main__":
    retrieve("validation_volume_9",
             "candidates",
             {"min":0, "max":250 * 5},
             {"min":0, "max":250* 5},
             {"min":0, "max":50*50},
             [5.,5.,50.],
             "/media/nilsec/d0/gt_mt_data/",
             distance_threshold=None)
