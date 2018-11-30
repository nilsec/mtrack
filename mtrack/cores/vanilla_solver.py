import os
import numpy as np

from mtrack.preprocessing import extract_candidates_double, candidates_to_g1,\
                                 connect_graph_locally, g1_to_nml
from mtrack.solve import solve


class VanillaSolver(object):

    def get_candidates(self,
                       prob_map_stack_chunk,
                       offset_chunk,
                       gs,
                       ps,
                       voxel_size,
                       id_offset):

        candidates = extract_candidates_double(prob_map_stack_chunk,
                                               gs,
                                               ps,
                                               voxel_size,
                                               bounding_box=None,
                                               bs_output_dir=None,
                                               offset_pos=offset_chunk,
                                               identifier_0=id_offset)

        return candidates

    def get_g1_graph(self,
                     candidates,
                     voxel_size):

        g1 = candidates_to_g1(candidates, 
                              voxel_size)

        return g1
        
    def connect_g1_graph(self,
                         g1, 
                         distance_threshold,
                         output_dir=None,
                         voxel_size=None):
        
        g1_connected = connect_graph_locally(g1,
                                             distance_threshold)

        if output_dir is not None:
            assert(voxel_size is not None)
            connected_dir = os.path.join(output_dir, "ccs")
            if not os.path.exists(connected_dir):
                os.makedirs(connected_dir)
                
            g1_connected.save(os.path.join(connected_dir, "connected.gt"))
            g1_to_nml(g1_connected,
                      os.path.join(connected_dir, "connected.nml"),
                      knossos=True,
                      voxel_size=voxel_size)
                                                
        return g1_connected

    def solve_g1_graph(self,
                       g1_connected,
                       cc_min_vertices,
                       start_edge_prior,
                       selection_cost,
                       distance_factor,
                       orientation_factor,
                       comb_angle_factor,
                       output_dir,
                       time_limit,
                       voxel_size):
        
        ccs = g1_connected.get_components(min_vertices=cc_min_vertices,
                                          output_folder=os.path.join(output_dir, "ccs/"),
                                          return_graphs=True)

        solutions = []
        j = 0
        for cc in ccs:
            cc.g.reindex_edges()
            cc_solution = solve(cc,
                                start_edge_prior,
                                distance_factor,
                                orientation_factor,
                                comb_angle_factor,
                                selection_cost,
                                time_limit,
                                output_dir=None,
                                voxel_size=None,
                                z_correction=0,
                                chunk_shift=np.array([0.,0.,0.]))

            self.save_solutions([cc_solution],
                                voxel_size,
                                output_dir,
                                n_0=j)
            j += 1
            
            solutions.append(cc_solution)

        return solutions

    def save_solutions(self,
                       solutions,
                       voxel_size,
                       output_dir,
                       n_0):

        solution_dir = os.path.join(output_dir, "solution")
    
        if not os.path.exists(solution_dir):
            os.makedirs(solution_dir)

        n = n_0
        for solution in solutions:
            g1_to_nml(solution, 
                      os.path.join(solution_dir, "cc_{}.nml".format(n)),
                      knossos=True,
                      voxel_size=voxel_size)

            solution.save(os.path.join(solution_dir, "cc_{}.gt".format(n)))
            n += 1

        
