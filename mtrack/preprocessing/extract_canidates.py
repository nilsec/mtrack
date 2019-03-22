import sys
import os
sys.path.append(os.path.join('..', '..'))

import h5py
import numpy as np
from mtrack.graphs import g1_graph
from scipy.spatial import KDTree
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


class MTCandidate(object):
    def __init__(self, position, orientation, identifier, partner_identifier=-1):
        self.position = position
        self.orientation = orientation
        self.identifier = identifier
        self.partner_identifier = partner_identifier

    def __repr__(self):
        return "<MtCandidateObject | position: %s, orientation: %s, id: %s, partner id: %s> \n" %\
        (self.position, self.orientation, self.identifier, self.partner_identifier)


def extract_maxima_candidates(maxima,
                              offset_pos=np.array([0,0,0]),
                              identifier_0=0):
    f = h5py.File(maxima, "r")
    maxima = np.array(f["maxima"])
    f.close()

    candidate_positions = np.array(np.nonzero(maxima)).T
    candidates = [MTCandidate(candidate_positions[i][::-1] + offset_pos, np.array([1.,0.,0.]), i + identifier_0, partner_identifier=-1) for i in range(len(candidate_positions))]
    
    return candidates


def candidates_to_g1(candidates, voxel_size):
    g1 = g1_graph.G1(len(candidates))
    
    id = 0
    for candidate in candidates:
        assert(candidate.identifier == id) # Partner derived from identifier

        position_phys = np.array([candidate.position[j] * voxel_size[j] for j in range(3)])
        orientation_phys = np.array([candidate.orientation[j] * voxel_size[j] for j in range(3)])
        partner = candidate.partner_identifier
        
        g1.set_position(id, position_phys)
        g1.set_orientation(id, orientation_phys)
        g1.set_partner(id, partner)
    
        id += 1

    return g1


def connect_graph_locally(g1, distance_threshold):
    positions = []
    id = 0
    for v in g1.get_vertex_iterator():
        positions.append(np.array(g1.get_position(v)))
        assert(v == id)
        id += 1

    kdtree = KDTree(positions)
    pairs = kdtree.query_pairs(distance_threshold, p=2.0, eps=0)

    
    for edge in pairs:
        if g1.get_partner(edge[0]) != edge[1]:
            """
            Only connect edges that have not been
            connected before. Can happen in context area.
            """
            try:
                e = g1.add_edge(*edge)
            except AssertionError:
                pass

    return g1
