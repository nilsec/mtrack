import sys
import os
sys.path.append(os.path.join('..', '..'))

import collections
import h5py
import numpy as np
from skimage.filters import gaussian
from skimage.measure import regionprops
from scipy import ndimage
import angle_estimate
from mt_utils import h5_tools
from graphs import g1_graph
import nml_io
import pickle
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

diam_out = 24 # Outer diameter of microtubule in nm

class DirectionType:
    def __init__(self, perpendicular, parallel):
        assert(type(perpendicular) == type(parallel))
        self.perp = perpendicular
        self.par = parallel
        self.index = 0

    def __iter__(self):
        return self

    def __repr__(self):
        return "(perp: " + str(self.perp) + ', par: ' + str(self.par) + ")"

    def next(self):
        if isinstance(self.perp, collections.Iterable):
            assert(len(self.perp) == len(self.par))
            
            if self.index == len(self.perp):
                self.index = 0
                raise StopIteration

            self.index += 1
            return DirectionType(self.perp[self.index-1], self.par[self.index-1])

        else:
            raise TypeError(str(type(self.perp)) + " is not iterable")


def process_bounding_box(bounding_box):
    if len(bounding_box) == 2:
        assert(isinstance(bounding_box[0], int))
        assert(isinstance(bounding_box[1], int))

        slices = range(bounding_box[0], bounding_box[1])
        x_lim = None
        y_lim = None

    else:
        xyz_bb = [[], [], []]

        for j in range(3):
            for corner in bounding_box:
                xyz_bb[j].append(corner[j])

        slices = range(int(min(xyz_bb[2])), int(max(xyz_bb[2])))
        x_lim = [int(min(xyz_bb[0]) - 0.5), int(max(xyz_bb[0]) + 0.5)]
        y_lim = [int(min(xyz_bb[1]) - 0.5), int(max(xyz_bb[1]) + 0.5)]

    return slices, x_lim, y_lim
 
    
def get_binary_stack(prob_map_stack_file, 
                     gaussian_sigma, 
                     point_threshold, 
                     voxel_size, 
                     verbose=False, 
                     bounding_box=None,
                     output_directory=None):
    """
    prob_map_stack_file: h5 stack containing slices of probability maps.
    """
    
    combined_binary_image_stack = []

    if isinstance(prob_map_stack_file, DirectionType):
        if verbose:
            print "Get binary stack, directional...\n"
    
        binary_image_stack = DirectionType(None, None)
        
        print "Get perpendicular binary stack..."
        binary_image_stack.perp = get_binary_stack(prob_map_stack_file.perp, 
                                                   gaussian_sigma.perp, 
                                                   point_threshold.perp, 
                                                   voxel_size, 
                                                   verbose=False, 
                                                   bounding_box=bounding_box,
                                                   output_directory=output_directory)

        print "Get parallel binary stack..."
        binary_image_stack.par = get_binary_stack(prob_map_stack_file.par, 
                                                  gaussian_sigma.par, 
                                                  point_threshold.par, 
                                                  voxel_size, 
                                                  verbose=False, 
                                                  bounding_box=bounding_box,
                                                  output_directory=output_directory)
   
        print "Combine slices..." 
        j = 0
        for binary_image in binary_image_stack:
            print "Combine slice {}".format(j)
            j += 1

            s_closing = np.ones((5,5))
            binary_image.par = ndimage.binary_closing(binary_image.par, s_closing).astype(int)
           
            s_label = ndimage.generate_binary_structure(2,2) 
            cc_par, n_features_par = ndimage.label(np.array(binary_image.par, dtype=np.uint32), 
                                                   structure=s_label) 

            # Filter connected components in parallel binary image that are smaller than the outer diameter of a mt (24 nm)
            for cc_label in xrange(1, cc_par.max() + 1):
                cc_masked = np.array(cc_par == cc_label, dtype=int) * cc_label # Mask connected component
                cc_features = regionprops(cc_masked, cache=True)
                assert(len(cc_features)==1)

                if (cc_features[0].major_axis_length * voxel_size[0]) <= diam_out:
                    cc_par -= cc_masked
    
            binary_image.par = np.array(cc_par > 0.5, dtype=int)    
            combined_binary_image_stack.append(np.array(np.logical_or(binary_image.par, 
                                                                      binary_image.perp), 
                                                                      dtype=int)
                                              )
        if output_directory is not None:
            h5_tools.stack_to_h5(np.transpose(combined_binary_image_stack),
                                 output_directory +\
                                 "bs_combined_gs-par-{}_".format(gaussian_sigma.par)+\
                                 "gs-perp-{}_".format(gaussian_sigma.perp)+\
                                 "ps-par-{}_".format(point_threshold.par)+\
                                 "ps-perp-{}".format(point_threshold.perp) + ".h5")
 

    else:
        if verbose:
            print "Get binary stack, simple...\n"
        f = h5py.File(prob_map_stack_file)
        prob_map_stack = f['exported_data'].value
        f.close()

        if bounding_box is not None:
            slices, x_lim, y_lim = process_bounding_box(bounding_box)

        else:
            slices = range(prob_map_stack.shape[2])
            x_lim = None
            y_lim = None

        for slice_id in slices:
            print "Process slice {}".format(slice_id)
            if x_lim is None:
                prob_map = prob_map_stack[:, :, slice_id]

            else:
                prob_map = prob_map_stack[y_lim[0]:y_lim[1], 
                                          x_lim[0]:x_lim[1], 
                                          slice_id]

            prob_map_smooth = gaussian(prob_map, gaussian_sigma)
            binary_image = prob_map_smooth > point_threshold
            combined_binary_image_stack.append(np.array(binary_image, dtype=int))
        
        if output_directory is not None:
            h5_tools.stack_to_h5(np.transpose(combined_binary_image_stack), 
                                 output_directory +\
                                 "bs_gs-{}_".format(gaussian_sigma)+\
                                 "ps-{}_".format(point_threshold) +\
                                 prob_map_stack_file.split("/")[-3] + ".h5")

    return combined_binary_image_stack


def extract_candidates(prob_map_stack_file, 
                       gaussian_sigma, 
                       point_threshold, 
                       voxel_size,
                       length_correction=0.0, 
                       verbose=False, 
                       bounding_box=None,
                       bs_output_dir=None):

    if verbose:
        print "\nExtract Candidates\n"
        print "Prob Map Stack File: ", prob_map_stack_file, "\n"
        print "Gaussian Sigma: %s \n" % gaussian_sigma
        print "Voxel Size: %s \n" % voxel_size

    binary_image_stack = get_binary_stack(prob_map_stack_file=prob_map_stack_file, 
                                          gaussian_sigma=gaussian_sigma, 
                                          point_threshold=point_threshold, 
                                          voxel_size=voxel_size, 
                                          verbose=verbose, 
                                          bounding_box=bounding_box,
                                          output_directory=bs_output_dir)

    if bounding_box is not None:
        slices, x_lim, y_lim = process_bounding_box(bounding_box)
        slice_number = slices[0]

    else:
        slice_number = 0

    identifier_0 = 0
    candidates = []


    if verbose:
        print "Process binary stack...\n"
    for binary_image in binary_image_stack:
        s_label = ndimage.generate_binary_structure(2,2)
        cc, n_features = ndimage.label(np.array(binary_image, dtype=np.uint32), 
                                       structure=s_label)
        
        candidate_list, max_identifier =\
            angle_estimate.get_candidates(cc=cc, 
                                          slice_number=slice_number, 
                                          voxel_x=voxel_size[0], 
                                          voxel_y=voxel_size[1], 
                                          voxel_z=voxel_size[2], 
                                          length_correction=length_correction, 
                                          identifier_0=identifier_0)

        identifier_0 = max_identifier
        
        candidates += candidate_list
     
        slice_number += 1 

    if bounding_box is not None:
        if x_lim is not None:
            for candidate in candidates:
                candidate.position = tuple(np.array(candidate.position) +\
                                       np.array((x_lim[0], y_lim[0], 0)))

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
    if g1.get_number_of_edges() != 0:
        raise Warning("G1 graph already connected")

    print "Construct KDTree...\n"

    positions = []
    id = 0
    for v in g1.get_vertex_iterator():
        positions.append(np.array(g1.get_position(v)))
        assert(v == id)
        id += 1

    kdtree = KDTree(positions)
    
    print "Query pairs...\n"
    # Can tweek eps for speed gains
    pairs = kdtree.query_pairs(distance_threshold, p=2.0, eps=0)

    print "Construct graph...\n"
    
    for edge in pairs:
        if g1.get_partner(edge[0]) != edge[1]:
            g1.add_edge(*edge)

    return g1


def get_distance_histogram(positions, n_samples):
    n_tot = len(positions)
    d_all_to_all = []

    ind_1 = np.random.randint(0, n_tot, size=n_samples)

    for i in ind_1:
        print str(i) + "/" + str(n_samples)
        for j in range(len(positions)):
            if j != i and not j in ind_1:
                d_all_to_all.append(np.linalg.norm(positions[i] - positions[j]))

    pickle.dump(d_all_to_all, open("./d_all_to_all", "wb"))

    plt.hist(d_all_to_all, normed=True)
    plt.show()

def chunk_volume(prob_map_stack_file, 
                 n_slices,
                 xy_fraction):

    f = h5py.File(prob_map_stack_file)
    prob_map_stack = f['exported_data'].value
    f.close()

    volume_dimensions = np.shape(prob_map_stack)

    x = volume_dimensions[0]
    y = volume_dimensions[1]
    z = volume_dimensions[2]

    mod = z % n_slices
 

    if mod != 0:
        raise Warning("N_slices % n_slices = {} % {} = {}".format(z, n_slices, mod))

    
    print volume_dimensions


if __name__ == "__main__":
    prob_map_stack_file_perp = "/media/nilsec/d0/gt_mt_data/" +\
                               "probability_maps/validation/perpendicular/stack/stack.h5"
 
    chunk_volume(prob_map_stack_file_perp, 10, 1.0)

    """
    prob_map_stack_file_perp = "/media/nilsec/d0/gt_mt_data/" +\
                               "probability_maps/validation/perpendicular/stack/stack.h5"

    prob_map_stack_file_par = "/media/nilsec/d0/gt_mt_data/" +\
                              "probability_maps/validation/parallel/stack/stack.h5" 
 
    prob_map_stack_file_direction = DirectionType(prob_map_stack_file_perp, prob_map_stack_file_par)

    base_dir = "/media/nilsec/d0/gt_mt_data/experiments/"

    for dist in [89]:
        candidates = pickle.load(open("candidates.p", "rb"))
        g1 = candidates_to_g1(candidates, voxel_size)
        g1_connected = connect_graph_locally(g1, dist)
        cc_list = g1_connected.get_components(min_vertices=4, 
                                              output_folder= base_dir + "cc_nopcedge_dist%s/" % dist,
                                              voxel_size=voxel_size)
    """
