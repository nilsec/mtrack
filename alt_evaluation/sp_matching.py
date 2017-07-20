import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import graphs
import evaluation

class GtVolume(object):
    def __init__(self):
        self.id_dict = {}

    def add_volume(self, volume, id):
        non_zero = np.nonzero(volume)
        non_zero = np.vstack([p for p in non_zero]).T

        for point in non_zero:
            # point = (z,y,x)
            try:
                self.id_dict[tuple(point)].append(id)
            except KeyError:
                self.id_dict[tuple(point)] = [id]

    def get_ids(self, point):
        try:
            return self.id_dict[point]
        except KeyError:
            return [0]

    def get_binary_volume(self, id, dimensions):
        canvas = np.zeros([dimensions[2], dimensions[1], dimensions[0]])

        for point, id_list in self.id_dict.iteritems():
            if id in id_list:
                canvas[point[0], point[1], point[2]] = 1

        return canvas
                

def get_gt_volume(gt_lines, dimensions, tolerance_radius, voxel_size, correction=np.array([0,0,0])):
    gt_volume = GtVolume()

    print "Interpolate Nodes..."
    label = 1
    for line in gt_lines:
        canvas = np.zeros([dimensions[2], dimensions[1], dimensions[0]])

        if isinstance(line, str):
            g1 = graphs.g1_graph.G1(0)
            g1.load(line)

        elif isinstance(line, graphs.g1_graph.G1):
            g1 = line

        else:
            raise TypeError("A line must be a path or a g1 graph.")


        for edge in g1.get_edge_iterator():
            start = np.array(g1.get_position(edge.source()), dtype=int)
            start -= correction
            end = np.array(g1.get_position(edge.target()), dtype=int)
            end -= correction

            dda = evaluation.DDA3(start, end)
            skeleton_edge = dda.draw()
            
            for point in skeleton_edge:
                canvas[point[2], point[1], point[0]] = 1

        line_volume = enlarge_binary_map(binary_map=canvas, 
                                         voxel_size=np.array([voxel_size[2], voxel_size[1], voxel_size[0]]),
                                         marker_size_physical=tolerance_radius)

        gt_volume.add_volume(line_volume, label)
        label += 1 

    return gt_volume


def enlarge_binary_map(binary_map, marker_size_voxel=1, voxel_size=None, marker_size_physical=None):
    """
    Enlarge existing regions in a binary map.
    Parameters
    ----------
        binary_map: numpy array
            matrix with zeros, in which regions to be enlarged are indicated with a 1 (regions can already
            represent larger areas)
        marker_size_voxel: int
            enlarged region have a marker_size (measured in voxels) margin added to
            the already existing region (taking into account the provided voxel_size). For instance a marker_size_voxel
            of 1 and a voxel_size of [2, 1, 1] (z, y, x) would add a voxel margin of 1 in x,y-direction and no margin
            in z-direction.
        voxel_size:     tuple, list or numpy array
            indicates the physical voxel size of the binary_map.
        marker_size_physical: int
            if set, overwrites the marker_size_voxel parameter. Provides the margin size in physical units. For
            instance, a voxel_size of [20, 10, 10] and marker_size_physical of 10 would add a voxel margin of 1 in
            x,y-direction and no margin in z-direction.
    Returns
    ---------
        binary_map: matrix with 0s and 1s of same dimension as input binary_map with enlarged regions (indicated with 1)
    """
    if voxel_size is None:
        voxel_size = (1, 1, 1)
    voxel_size = np.asarray(voxel_size)
    if marker_size_physical is None:
        voxel_size /= np.min(voxel_size)
        marker_size = marker_size_voxel
    else:
        marker_size = marker_size_physical
    binary_map = np.logical_not(binary_map)
    binary_map = distance_transform_edt(binary_map, sampling=voxel_size)
    binary_map = binary_map <= marker_size
    binary_map = binary_map.astype(np.uint8)
    return binary_map
