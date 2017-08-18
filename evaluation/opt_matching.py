import pylp
import pdb
import numpy as np
from scipy.spatial import KDTree

def is_arr_in_list(arr, arr_list):
    return next((True for elem in arr_list if elem is arr), False)

class OptMatch(object):
    def __init__(self, lines_gt, lines_rec, n, distance_tolerance, voxel_size=np.array([5.0,5.0,50.0])):
        assert(isinstance(lines_gt, list))
        assert(isinstance(lines_rec, list))

        self.lines_gt = lines_gt
        self.lines_rec = lines_rec
        self.n = n

        self.gt_chunks, self.gt_chunk_positions, self.inv_gt_chunk_positions, n_lines\
            = self.get_chunks(self.lines_gt, self.n)

        self.rec_chunks, self.rec_chunk_positions, self.inv_rec_chunk_positions, _\
            = self.get_chunks(self.lines_rec, self.n, l_0=n_lines)

        chunk_positions = self.rec_chunk_positions.values() + self.gt_chunk_positions.values() 

        self.pairs = self.connect_chunks(chunk_positions,
                                         voxel_size, 
                                         distance_tolerance)

        self.gt_rec_pairs = []

        for pair in self.pairs:
            pos_0 = chunk_positions[pair[0]] 
            pos_1 = chunk_positions[pair[1]] 
            if is_arr_in_list(pos_0, self.rec_chunk_positions.values()):
                if is_arr_in_list(pos_1, self.gt_chunk_positions.values()):
                    self.gt_rec_pairs.append((self.inv_rec_chunk_positions[tuple(pos_0)],
                                              self.inv_gt_chunk_positions[tuple(pos_1)]))

            else:
                if is_arr_in_list(pos_1, self.rec_chunk_positions.values()):
                    self.gt_rec_pairs.append((self.inv_rec_chunk_positions[tuple(pos_1)],
                                              self.inv_gt_chunk_positions[tuple(pos_0)]))

        self.edge_pairs, self.edge_pair_cost = self.get_edge_pairs(self.gt_rec_pairs)
        pdb.set_trace()
        
        self.backend = pylp.GurobiBackend()
        self.backend.initialize(len(self.rec_chunks.values() + self.gt_chunks.values()), 
                                    pylp.VariableType.Binary)
        self.backend.initialize(len(self.gt_rec_pairs), pylp.VariableType.Binary)
        self.backend.initialize(len(self.edge_pairs), pylp.VariableType.Binary)    
    
    
    def get_chunks(self, lines, n, l_0=0):
        chunks = {}
        chunk_positions = {}
        inv_chunk_positions = {}

        l = l_0
        for line in lines:

            k = 0
            chunks[(l, k)] = []
            pos = np.array([0,0,0])

            m = 0
            for voxel in line:
                chunks[(l, k)].append(voxel)
                pos += voxel
                m += 1

                if not m % n:
                    chunk_pos = pos/float(n)
                    chunk_positions[(l, k)] = chunk_pos
                    inv_chunk_positions[tuple(chunk_pos)] = (l,k)
                    k += 1
                    chunks[l,k] = []
                    pos = np.array([0,0,0])

                if m == len(line) and m % n:
                    chunk_pos = pos/(m%n)

                    chunk_positions[(l, k)] = chunk_pos
                    inv_chunk_positions[tuple(chunk_pos)] = (l,k)
            l += 1
 
        return chunks, chunk_positions, inv_chunk_positions, l

    def connect_chunks(self, chunk_positions, voxel_size, distance_tolerance):
        kdtree = KDTree(chunk_positions * voxel_size)
        pairs = kdtree.query_pairs(distance_tolerance, p=2.0, eps=0)
       
        return pairs

    def get_edge_pairs(self, gt_rec_pairs):
        N_edges = len(gt_rec_pairs)
        edge_pairs = []
        edge_pair_cost = []

        for i in range(N_edges):
            for j in range(N_edges):
                if j>i:
                    edge_i = gt_rec_pairs[i]
                    edge_j = gt_rec_pairs[j]

                    edge_i_lines = [edge_i[0][0], edge_i[1][0]]
                    edge_j_lines = [edge_j[0][0], edge_j[1][0]]

                    match = None
                    line_i = 0
                    for i_id in edge_i_lines:
                        line_j = 0
                        for j_id in edge_j_lines:
                            if i_id == j_id:
                                match = (line_i, line_j)
                            line_j += 1
                        line_i += 1
                                
                    if match is not None:
                        if abs(edge_i[match[0]][1] - edge_j[match[1]][1]) == 1:
                            edge_pairs.append((edge_i, edge_j))
                            if edge_i[int(not match[0])][0] != edge_j[int(not match[1])][0]:
                                edge_pair_cost.append(1)
                            else:
                                edge_pair_cost.append(0)
            
        return edge_pairs, edge_pair_cost
