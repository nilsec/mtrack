import graphs
import json
import os
import numpy as np
from process_solution import get_lines
import pdb
from scipy.spatial.distance import cdist
import h5py
import preprocessing

def get_intersect(x,y):
    return range(max(x[0], y[0]), min(x[-1], y[-1]) + 1)

class Stitcher(object):

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
            line_graph_1 = graphs.G1(0)
            line_graph_1.load(line_1)

            positions_1 = line_graph_1.get_position_array()

            z_match = max(position_1[2]) in z_intersect or min(position_1[2]) in z_intersect
            y_match = max(position_1[1]) in y_intersect or min(position_1[1]) in y_intersect
            x_match = max(position_1[0]) in x_intersect or min(position_1[0]) in x_intersect

            overlap = z_match and y_match and x_match 
 
            if not overlap:
                n_line += 1
                continue
 
            for line_2 in chunk_2_lp:
                line_graph_2 = graphs.G1(0)
                line_graph_2.load(line_2)
                matches[line_1] = (line_2, 0)
                positions_2 = line_graph_2.get_position_array()
                
                if not reached_line_2:
                    z_match = max(position_2[2]) in z_intersect or min(position_2[2]) in z_intersect
                    y_match = max(position_2[1]) in y_intersect or min(position_2[1]) in y_intersect
                    x_match = max(position_2[0]) in x_intersect or min(position_2[0]) in x_intersect
                    
                    overlap = z_match and y_match and x_match
 
                
                    if not overlap:
                        chunk_2_lp.remove(line_2)
                        continue

                d_pair = cdist(positions_1.T,
                               positions_2.T)
            
                n_matches = np.sum(d_pair < d)
                if matches[line_1][1] < n_matches:
                    matches[line_1] = (line_2, n_matches)
                
            n_line += 1
            reached_line_2  = True

        k = 0
        for match, matcher in matches.iteritems():
            if matcher[1] > n_min:
                preprocessing.g1_to_nml(match, 
                                        output_dir + "/matches/match_1_{}.nml".format(k),
                                        knossify=True)

                preprocessing.g1_to_nml(matcher[0], 
                                        output_dir + "/matches/match_2_{}.nml".format(k),
                                        knossify=True)

                k += 1
                

        json.dump(matches, open("./matches.json", "w"))
