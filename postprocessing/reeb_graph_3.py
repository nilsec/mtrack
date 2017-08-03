from graphs.graph import G
import operator
import evaluation
import numpy as np
import os
import graphs
import cPickle as pickle
from scipy.ndimage.morphology import distance_transform_edt
import pdb


class ReebGraph(G):
    def __init__(self, gt_line_dir):
        #Init Graph Structure
        G.__init__(self, 0, None)
        G.new_vertex_property(self, "id", dtype="int")
        G.new_vertex_property(self, "pos", dtype="vector<int>")

        #Raw ground truth lines in graph tools format
        self.gt_lines = [os.path.join(gt_line_dir, f) for f in os.listdir(gt_line_dir) if f.endswith(".gt")]

    def build_graph(self, dimensions, correction, epsilon, save_canvas_dict=None, load_canvas_dict=None):
        if load_canvas_dict is None:
            canvas = np.zeros([dimensions[2], dimensions[1], dimensions[0]])

            canvas_dict = {}
            line_dict = {n: [] for n in range(1, len(self.gt_lines) + 1)}

            label = 1
            for line in self.gt_lines:
                g1 = graphs.g1_graph.G1(0)
                g1.load(line)

                for edge in g1.get_edge_iterator():
                    start = np.array(g1.get_position(edge.source()), dtype=int)
                    start -= correction
                    end = np.array(g1.get_position(edge.target()), dtype=int)
                    end -= correction

                    dda = evaluation.DDA3(start, end)
                    skeleton_edge = dda.draw()
                    line_dict[label].extend(skeleton_edge)
            
                    for point in skeleton_edge:
                        canvas[point[2], point[1], point[0]] = 1

                canvas = enlarge_binary_map(canvas, 
                                            marker_size_voxel=1, 
                                            voxel_size=None, 
                                            marker_size_physical=epsilon, 
                                            load_distance_transform=None,
                                            save_distance_transform=None)  

                canvas_dict[label] = (canvas, line)
                label += 1

            if save_canvas_dict is not None:
                if not os.path.exists(os.path.dirname(save_canvas_dict)):
                    os.makedirs(os.path.dirname(save_canvas_dict))
                pickle.dump(canvas_dict, open(save_canvas_dict, "w+"))
        else:
            print "Load Canvas Dict..."
            canvas_dict = pickle.load(open(load_canvas_dict, "r"))

        build_z_graphs(canvas_dict, dimensions)

def build_z_graphs(canvas_dict, dimensions):
    z_graphs = [ZGraph(z) for z in range(dimensions[2])]
    match_volume = np.vstack([val[0] for val in canvas_dict.values()])
    match_volume_ids = canvas_dict.keys()

    dict_length = len(canvas_dict)
    for roll in range(dict_length):
        print roll, "/", dict_length

        match_canvas = np.roll(match_volume, roll, axis=0)
        match_canvas_ids = np.roll(match_volume_ids, roll)
        land = np.logical_and(match_canvas, match_volume)
        nz_z = np.nonzero(land)[0]
        
        z_matches = nz_z % 2
        id_matches = nz_z / 2

        for n, z in enumerate(z_matches):
            id_ = id_matches[n]
            label_i = match_volume_ids[id_]
            label_j = match_canvas_ids[id_]

            z_graphs[z].add_z_vertex(label_i)
            z_graphs[z].add_z_vertex(label_j)
            z_graphs[z].add_z_edge(label_i, label_j)

    
    for z in z_graphs:
        z.save("/media/nilsec/d0/gt_mt_data/experiments/z_graphs/z_{}".format(z.z))


class ZGraph(G):
    def __init__(self, z):
        self.z = z
        G.__init__(self, 0, None)
        G.new_vertex_property(self, "id", dtype="int")
        self.vertex_ids = []

    def add_z_vertex(self, id_):
        if not (id_ in self.vertex_ids):
            self.vertex_ids.append(id_)
            v = G.add_vertex(self)
            G.set_vertex_property(self, "id", v, id_)

    def get_z_vertex(self, id_):
        v_id = None
        for v in G.get_vertex_iterator(self):
            if G.get_vertex_property(self, "id")[v] == id_:
                v_id = v
        if v_id is None:
            raise KeyError("Vertex id not in graph")
        
        return v_id

    def get_id(self, vertex):
        return G.get_vertex_property(self, "id")[vertex]

    def add_z_edge(self, id1, id2):
        if id1 != id2:
            G.add_edge(self, self.get_z_vertex(id1), self.get_z_vertex(id2))

    def get_connected_components(self):
        masks, hist = G.get_component_masks(self)
        ccs = []
        for mask in masks:
            G.set_vertex_filter(self, mask)
            id_set = set()

            for v in G.get_vertex_iterator(self):
                id_set.add(self.get_id(v))

            ccs.append(id_set)
            G.set_vertex_filter(self, None)

        print self.z, ccs
        return ccs

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        pickle.dump(self.__dict__, open(path, "w+"))

    def load(self, path):
        me = pickle.load(open(path, "r"))
        self.__dict__ = me


def enlarge_binary_map(binary_map, 
                       marker_size_voxel=1, 
                       voxel_size=None, 
                       marker_size_physical=None, 
                       load_distance_transform=None,
                       save_distance_transform=None):
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

    if load_distance_transform is None:
        binary_map = np.logical_not(binary_map)
        binary_map = distance_transform_edt(binary_map, sampling=voxel_size)
        if save_distance_transform is not None:
            pickle.dump(binary_map, open(save_distance_transform, "w+"))
    else:
        binary_map = pickle.load(load_distance_transform)

    binary_map = binary_map <= marker_size
    binary_map = binary_map.astype(np.uint8)
    return binary_map

if __name__ == "__main__":
    rg = ReebGraph("/media/nilsec/d0/gt_mt_data/experiments/sp_test/trace_small_lines")
    output_dir = "/media/nilsec/d0/gt_mt_data/experiments/reeb_graph_test"
    rg.build_graph(dimensions=[1025, 1025, 21], 
                   correction=[0,0,300], 
                   epsilon=50, 
                   load_canvas_dict=output_dir + "/canvas_dict.p")
            
             
