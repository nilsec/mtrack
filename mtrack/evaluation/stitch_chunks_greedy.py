import numpy as np
import os
import json
import h5py
from scipy.spatial.distance import cdist
import copy

from mtrack.graphs import G1
import mtrack.preprocessing
from mtrack.evaluation.process_solution import get_lines
from mtrack.postprocessing.combine_solutions import combine_gt_graphs

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
                      voxel_size):

        combined_graph = combine_gt_graphs([chunk_1_solution, chunk_2_solution])
        
        #Get nodes in overlap region:
        positions = combined_graph.get_position_array()
        
        pdb.set_trace()

    
    def get_overlap_vertices(self, 
                             chunk_1_solution,
                             chunk_2_solution,
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

        p1 = chunk_1_solution.get_position_array()
        p2 = chunk_2_solution.get_position_array()

        # Check which points are in overlap region
        isin1_x = np.in1d(p1[0], x_intersect)
        isin1_y = np.in1d(p1[1], y_intersect)
        isin1_z = np.in1d(p1[2], z_intersect)

        isin2_x = np.in1d(p2[0], x_intersect)
        isin2_y = np.in1d(p2[1], y_intersect)
        isin2_z = np.in1d(p2[2], z_intersect)

        isin1 = np.all(np.stack([isin1_x, isin1_y, isin1_z]), axis=0)
        isin2 = np.all(np.stack([isin2_x, isin2_y, isin2_z]), axis=1)

        # Mask out subgraph in overlap region
        chunk_1_solution.set_vertex_mask(isin1)
        chunk_2_solution.set_vertex_mask(isin2)

        # Find 

        

        

        

        
        


        
         
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
