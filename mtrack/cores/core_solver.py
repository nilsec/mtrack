import numpy as np

from mtrack.solve import solve

class CoreSolver(object):

    def check_forced(self, g1):
        """
        Check that the number of forced egdes
        incident to any vertex is <= 2 for 
        a given g1 graph.
        """
        for v in g1.get_vertex_iterator():
            incident = g1.get_incident_edges(v)
            forced = [g1.get_edge_property("selected", u=e.source(), v=e.target()) for e in incident]
            assert(sum(forced)<=2)


    def solve_subgraph(self, 
                       subgraph,
                       index_map,
                       cc_min_vertices,
                       start_edge_prior,
                       selection_cost,
                       orientation_factor,
                       distance_factor,
                       comb_angle_factor,
                       core_id,
                       voxel_size,
                       time_limit,
                       backend="Gurobi"):


        print "Solve connected subgraphs..."
        ccs = subgraph.get_components(min_vertices=cc_min_vertices,
                                      output_folder=None,
                                      return_graphs=True)
        
        j = 0
        solutions = []
        for cc in ccs:
            cc.reindex_edges_save()
            self.check_forced(cc)

            cc_solution = solve(cc,
                                start_edge_prior,
                                orientation_factor,
                                distance_factor,
                                comb_angle_factor,
                                selection_cost,
                                time_limit,
                                output_dir=None,
                                voxel_size=None,
                                chunk_shift=np.array([0.,0.,0.]),
                                backend=backend)

            solutions.append(cc_solution)

            j += 1

        return solutions
