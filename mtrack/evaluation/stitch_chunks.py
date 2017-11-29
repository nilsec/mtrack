import numpy as np
import os
import json
import h5py
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import copy

from mtrack.graphs import G1
import mtrack.preprocessing
from mtrack.evaluation.process_solution import get_lines
from mtrack.postprocessing.combine_solutions import combine_gt_graphs
from mtrack.solve import solve_volume
from mtrack.preprocessing import g1_to_nml

import pdb

def get_intersect(x,y):
    return range(max(x[0], y[0]), min(x[-1], y[-1]) + 1)

class Stitcher(object):
    def match_optimal(self, 
                      chunk_1_solution,
                      chunk_2_solution,
                      chunk_1,
                      chunk_2,
                      d,
                      voxel_size,
                      output_dir):

        if isinstance(chunk_1_solution, str):
            chunk_1_graph = G1(0)
            chunk_1_graph.load(chunk_1_solution)
            chunk_1_graph.g.purge_vertices()

        else:
            chunk_1_graph = chunk_1_solution
            chunk_1_graph.g.purge_vertices()
            chunk_1_graph.g.purge_edges()

        if isinstance(chunk_2_solution, str):
            chunk_2_graph = G1(0)
            chunk_2_graph.load(chunk_2_solution)

        else:
            chunk_2_graph = chunk_2_solution

        pdb.set_trace()
        # Label vertices for later identification in combined graph
        chunk_1_graph.new_vertex_property("chunk_id", "int", value=np.array([1] * chunk_1_graph.get_number_of_vertices()))
        chunk_2_graph.new_vertex_property("chunk_id", "int", value=np.array([2] * chunk_2_graph.get_number_of_vertices()))
        
        # Merge graphs
        combined_graph = combine_gt_graphs([chunk_1_graph, chunk_2_graph], prop_vp="chunk_id", prop_vp_dtype="int")
        
        g1_to_nml(combined_graph,
                  "./combined_graph.nml",
                  knossos=True,
                  voxel_size=voxel_size)
        
        # Mask out everything except overlap region
        masked_graph = self.mask_overlap_graph(combined_graph,
                                               chunk_1,
                                               chunk_2,
                                               d,
                                               voxel_size)
        g1_to_nml(masked_graph,
                 "./masked_graph.nml",
                 knossos=True,
                 voxel_size=voxel_size)

        masked_positions = masked_graph.get_position_array().T
        print "Found {} vertices in stitch region".format(np.shape(masked_positions)[0])

        # Get vertex ids corresponding to positions:
        index_map = {j: v for j,v in enumerate(masked_graph.get_vertex_iterator())}

        # Connect vertices in distance threshold if they belong to different chunks
        kdtree = KDTree(masked_positions)
        pairs = kdtree.query_pairs(d, p=2.0, eps=0)

        for edge in pairs:
            # Map edge ids
            edge = [ index_map[edge[0]], index_map[edge[1]] ]

            chunk_id_1 = masked_graph.get_vertex_property("chunk_id", edge[0])
            chunk_id_2 = masked_graph.get_vertex_property("chunk_id", edge[1])
            
            if chunk_id_1 != chunk_id_2:
                pos_1 = masked_graph.get_position(edge[0])
                pos_2 = masked_graph.get_position(edge[1])
                masked_graph.add_edge(*edge)


        masked_graph.g.clear_filters()

        g1_to_nml(masked_graph,
                  "./masked_graph_pre.nml",
                  knossos=True,
                  voxel_size=voxel_size)


        # Resolve on whole graph
        
        cc_list = masked_graph.get_components(min_vertices=2,
                                              output_folder = output_dir + "ccs/")

        g1_solution = solve_volume(output_dir + "ccs/",
                     start_edge_prior=160.0,
                     distance_factor=0.0,
                     orientation_factor=15.0,
                     comb_angle_factor=16.0,
                     selection_cost=-70.0,
                     time_limit=100,
                     output_dir = output_dir,
                     voxel_size =[5.,5.,50.] ,
                     combine_solutions=True)


        return g1_solution        

    
    def mask_overlap_graph(self, 
                           combined_graph,
                           chunk_1,
                           chunk_2,
                           d,
                           voxel_size):

        f = h5py.File(chunk_1, "r")
        attrs_1 = f["exported_data"].attrs.items()
        f.close()
        
        f = h5py.File(chunk_2, "r")
        attrs_2 = f["exported_data"].attrs.items()
        f.close()

        limits_1 = attrs_1[1][1]
        limits_2 = attrs_2[1][1]

        x_intersect = get_intersect(limits_1[0], limits_2[0])
        y_intersect = get_intersect(limits_1[1], limits_2[1])
        z_intersect = get_intersect(limits_1[2], limits_2[2])
        
        pos = (combined_graph.get_position_array().T/voxel_size).T.astype(int)

        # Check which points are in overlap region
        isin_x = np.in1d(pos[0], x_intersect)
        isin_y = np.in1d(pos[1], y_intersect)
        isin_z = np.in1d(pos[2], z_intersect)

        isin = np.all(np.stack([isin_x, isin_y, isin_z]), axis=0)
        
        # Mask out subgraph in overlap region
        combined_graph.set_vertex_mask(isin)

        return combined_graph

         
    def match_chunks(self, 
                     chunk_1_solution, 
                     chunk_2_solution, 
                     chunk_1,
                     chunk_2,
                     d,
                     n_min,
                     voxel_size,
                     output_dir):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        chunk_1_lp = get_lines(chunk_1_solution,
                               os.path.join(output_dir, "chunk_1/lines/"),
                               nml=False)

        chunk_2_lp = get_lines(chunk_2_solution,
                               os.path.join(output_dir, "chunk_2/lines/"),
                               nml=False)

        f = h5py.File(chunk_1, "r")
        attrs_1 = f["exported_data"].attrs.items()
        f.close()
        
        f = h5py.File(chunk_2, "r")
        attrs_2 = f["exported_data"].attrs.items()
        f.close()

        limits_1 = attrs_1[1][1]
        limits_2 = attrs_2[1][1]

        x_intersect = get_intersect(limits_1[0], limits_2[0])
        y_intersect = get_intersect(limits_1[1], limits_2[1])
        z_intersect = get_intersect(limits_1[2], limits_2[2])


        matches = {}
        n_line = 0
        reached_line_2 = False
        for line_1 in chunk_1_lp:
            print "Match line {}/{}".format(n_line + 1, len(chunk_1_lp))
            line_graph_1 = G1(0)
            line_graph_1.load(line_1)

            positions_1 = line_graph_1.get_position_array()

            z_match = max(positions_1[2]) in z_intersect or min(positions_1[2]) in z_intersect
            y_match = max(positions_1[1]) in y_intersect or min(positions_1[1]) in y_intersect
            x_match = max(positions_1[0]) in x_intersect or min(positions_1[0]) in x_intersect

            overlap = z_match and y_match and x_match 
 
            if not overlap:
                n_line += 1
                continue
            matches[line_1] = [(None, 0)]
 
            for line_2 in chunk_2_lp:
                line_graph_2 = G1(0)
                line_graph_2.load(line_2)
                positions_2 = line_graph_2.get_position_array()
                
                if not reached_line_2:
                    z_match = max(positions_2[2]) in z_intersect or min(positions_2[2]) in z_intersect
                    y_match = max(positions_2[1]) in y_intersect or min(positions_2[1]) in y_intersect
                    x_match = max(positions_2[0]) in x_intersect or min(positions_2[0]) in x_intersect
                    
                    overlap = z_match and y_match and x_match
                
                    if not overlap:
                        chunk_2_lp.remove(line_2)
                        continue

                d_pair = cdist(positions_1.T * np.array(voxel_size),
                               positions_2.T * np.array(voxel_size))
            
                n_matches = np.sum(d_pair < d)
                if n_matches > 0:
                    matches[line_1].append((line_2, n_matches))
                
            n_line += 1
            reached_line_2  = True

        matches_max = [sorted(m, key=lambda x: x[1], reverse=True) for m in matches.values()]

        k = 0
        for match, matcher in zip(matches.keys(), matches_max):
            j = 0
            for m in matcher:
                if m[1] > n_min:
                    if j == 0:
                        mtrack.preprocessing.g1_to_nml(match, 
                                               output_dir + "/matches/matchy/match_{}_{}.nml".format(k,j),
                                               knossify=True)

                    mtrack.preprocessing.g1_to_nml(m[0], 
                                                   output_dir + "/matches/matcher/match_{}_{}.nml".format(k,j),
                                                   knossify=True)
                    j += 1

            k += 1
                

        json.dump(matches, open("./matches.json", "w"))
