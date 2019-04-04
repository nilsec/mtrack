import sys
import os
sys.path.append(os.path.join('..', '..'))

import h5py
import numpy as np
from mtrack.graphs import g1_graph
from scipy.spatial import KDTree
from scipy import ndimage
from skimage.measure import regionprops


class MTCandidate(object):
    def __init__(self, position, orientation, identifier, partner_identifier=-1):
        self.position = position
        self.orientation = orientation
        self.identifier = identifier
        self.partner_identifier = partner_identifier

    def __repr__(self):
        return "<MtCandidateObject | position: %s, orientation: %s, id: %s, partner id: %s> \n" %\
        (self.position, self.orientation, self.identifier, self.partner_identifier)


def extract_cc_candidates(prob_map,
                          prob_map_dset,
                          threshold,
                          offset_pos=np.array([0,0,0]),
                          identifier_0=0):

    f = h5py.File(prob_map, "r")
    prob_map = np.array(f[prob_map_dset])
    f.close()

    binary_stack = []
    candidates = []
    i = 0
    for z in range(np.shape(prob_map)[0]):
        binary_map = prob_map[z,:,:] > threshold
        s_closing = np.ones((5,5))
        binary_map = ndimage.binary_closing(binary_map, s_closing).astype(int)
               
        s_label = ndimage.generate_binary_structure(2,2) 
        cc, n_features = ndimage.label(np.array(binary_map, dtype=np.uint32), structure=s_label)
        centroids = fit_ellipse(cc)
        for centroid in centroids:
            candidate = MTCandidate(np.array(centroid + [z]) + offset_pos, 
                                    np.array([1.,0.,0.]),
                                    i + identififer_0,
                                    partner_identifier=-1)
            candidates.append(candidate)
            i += 1

    return candidates


def fit_ellipse(cc, verbose=False, plot=False):
    """
    Fits an ellipse to the connected components of a binary image and returns major, minor axis, angle to x axis 
    as well as the centroid of each connected component.
    
    Parameters:
    ---------------------
    cc: Labeled input image. I.e. cc contains a matrix where connected components are indicated by different values.
        This is readily available as output from ndimage.label(). Labels with value zero are ignored.
    
    Returns:
    --------------------
    centroids
    """

    props = skimage.measure.regionprops(cc, cache=True)
    cc_features = []
    centroids = []
    centroid.append((feature[3], feature[4]))
    for prop in props:
        y_0, x_0 = prop.centroid 
        centroids.append([x0, y0])

    return centroids


def extract_maxima_candidates(maxima,
                              maxima_dset,
                              offset_pos=np.array([0,0,0]),
                              identifier_0=0):
    f = h5py.File(maxima, "r")
    maxima = np.array(f[maxima_dset])
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
