from mtrack.cores import CoreSolver
from mtrack.preprocessing import g1_to_nml, connect_graph_locally
import os

def retrieve(name_db,
             collection,
             x_lim,
             y_lim,
             z_lim,
             voxel_size,
             output_path):

    solver = CoreSolver()
    vertices, edges = solver.get_subgraph(name_db,
                                          collection,
                                          x_lim=x_lim,
                                          y_lim=y_lim,
                                          z_lim=z_lim)

    g1, index_map = solver.subgraph_to_g1(vertices, edges)

    g1_to_nml(g1, 
              output_path,
              knossos=True,
              voxel_size=voxel_size)

if __name__ == "__main__":
    retrieve("cremi_validation_a0_1",
             "microtubules",
             {"min":350*5, "max":1050 * 5},
             {"min":350*5, "max":1050* 5},
             {"min":7*50, "max":47 * 50},
             [5.,5.,50.],
             "/media/nilsec/d0/gt_mt_data/mtrack/grid_A+/grid_1/roi.nml")
