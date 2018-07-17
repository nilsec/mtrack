import numpy as np
import os
import json
from timeit import default_timer as timer
from shutil import copyfile
from copy import deepcopy
import glob
import h5py
from pymongo import MongoClient, IndexModel, ASCENDING
from scipy.spatial import KDTree
import itertools
from functools import partial
import pdb

from mtrack.graphs import g1_graph
from mtrack.preprocessing import extract_candidates, connect_graph_locally
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
                       distance_threshold,
                       cc_min_vertices,
                       start_edge_prior,
                       selection_cost,
                       distance_factor,
                       orientation_factor,
                       comb_angle_factor,
                       core_id,
                       voxel_size,
                       time_limit,
                       backend="Gurobi"):


        print "Solve connected subgraphs..."
        ccs = subgraph.get_components(min_vertices=cc_min_vertices,
                                      output_folder="./ccs/",
                                      return_graphs=True)

        j = 0
        solutions = []
        for cc in ccs:
            cc.reindex_edges_save()
            self.check_forced(cc)

            cc_solution = solve(cc,
                                start_edge_prior,
                                distance_factor,
                                orientation_factor,
                                comb_angle_factor,
                                selection_cost,
                                time_limit,
                                output_dir=None,
                                voxel_size=None,
                                chunk_shift=np.array([0.,0.,0.]),
                                backend=backend)

            for v in cc_solution.get_vertex_iterator():
                assert(len(cc_solution.get_incident_edges(v)) <= 2)
 
            solutions.append(cc_solution)

            j += 1

        return solutions
