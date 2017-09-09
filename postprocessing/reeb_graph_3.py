from graphs.graph import G
import operator
import evaluation
import numpy as np
import os
import graphs
import cPickle as pickle
from scipy.ndimage.morphology import distance_transform_edt
from scipy.sparse import *
import pdb
from scipy.spatial import KDTree

def is_arr_in_list(arr, arr_list):
    return next((True for elem in arr_list if elem is arr), False)

class ReebGraph(G):
    def __init__(self, gt_line_dir):
        #Init Graph Structure
        G.__init__(self, 0, None)
        G.new_vertex_property(self, "id", dtype="int")
        G.new_vertex_property(self, "id_set", dtype="vector<int>")
        G.new_vertex_property(self, "pos", dtype="vector<int>")

        #Raw ground truth lines in graph tools format
        self.gt_lines = [os.path.join(gt_line_dir, f) for f in os.listdir(gt_line_dir) if f.endswith(".gt")]

    def get_z_graphs(self, dimensions, correction, epsilon, voxel_size=np.array([5.,5.,50.])):
        assert(isinstance(voxel_size, np.ndarray))
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

            label += 1

        points = []
        [points.extend(j) for j in line_dict.values()]
        kdtree = KDTree(points * voxel_size)

        pairs = kdtree.query_pairs(2 * epsilon, p=2.0, eps=0)
        z_graphs = [ZGraph(z) for z in range(dimensions[2])]

        n = 1
        n_pairs = float(len(pairs))
        perc = 0.1
        for pair in pairs:
            if n/n_pairs > perc:
                print perc * 100, "%"
                perc += 0.1

            point_0 = points[pair[0]]
            point_1 = points[pair[1]]

            if point_0[2] != point_1[2]:
                n += 1
                continue
            else:
                z = point_0[2] # = point_1[2]
                
                vertices_p0 = []
                vertices_p1 = []
                for line_id, line_points in line_dict.iteritems():
                    if is_arr_in_list(point_0, line_points):
                        v0 = z_graphs[z].add_z_vertex(line_id)
                        if v0 is not None:
                            vertices_p0.append(v0)
                            #G.set_vertex_property(self, "pos", v0, point_0)
                            
 
                    if is_arr_in_list(point_1, line_points):
                        v1 = z_graphs[z].add_z_vertex(line_id)
                        if v1 is not None:
                            vertices_p1.append(v1)
                            #G.set_vertex_property(self, "pos", v1, point_1)
 
                    for v0 in vertices_p0:
                        for v1 in vertices_p1:
                            z_graphs[z].add_edge(v0, v1)

            n += 1

        for g in z_graphs:
            g.save("/media/nilsec/d0/gt_mt_data/experiments/z_graphs_200/z_{}".format(g.z))

    def build_graph(self, z_graphs):
        id_range = range(1, len(self.gt_lines) + 1)

        for line_id in id_range:
            for z_graph in z_graphs:
                if z_graph.ccs is None:
                    if z_graph.vertex_ids:
                        z_graph.get_connected_components()
                    else:
                        continue
                    
                    for id_set in z_graph.ccs:
                        if line_id in id_set:
                            v = G.add_vertex(self)
                            G.set_vertex_property(self, "id", line_id)
                
    
             

             
            
            
            
            
        
        return 0
                
                
class ZGraph(G):
    def __init__(self, z):
        self.z = z
        G.__init__(self, 0, None)
        G.new_vertex_property(self, "id", dtype="int")
        G.new_vertex_property(self, "pos", dtype="vector<int>")
        self.vertex_ids = []
        self.ccs = None

    def add_z_vertex(self, id_):
        if not (id_ in self.vertex_ids):
            self.vertex_ids.append(id_)
            v = G.add_vertex(self)
            G.set_vertex_property(self, "id", v, id_)
            return v
        return None

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

    def add_z_edge(self, v1, v2):
        if v1 != v2:
            G.add_edge(self, v1, v2)

    def add_z_edge_by_id(self, id1, id2):
        if id1 != id2:
            G.add_edge(self, self.get_z_vertex(id1), self.get_z_vertex(id2))

    def get_connected_components(self):
        try:
            masks, hist = G.get_component_masks(self)
        except:
            pdb.set_trace()
        ccs = []
        for mask in masks:
            G.set_vertex_filter(self, mask)
            id_set = set()

            for v in G.get_vertex_iterator(self):
                id_set.add(self.get_id(v))

            ccs.append(id_set)
            G.set_vertex_filter(self, None)

        print self.z, ccs
        self.ccs = ccs
        return ccs

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        pickle.dump(self.__dict__, open(path, "w+"))

    def load(self, path):
        me = pickle.load(open(path, "r"))
        self.__dict__ = me

def process_z_graphs(z_folder):
    z_list = os.listdir(z_folder)
    z_graph_list = []
    for path in z_list:
        z_graph = ZGraph(0)
        z_graph.load(os.path.join(z_folder, path))
        if z_graph.z != 20:
            z_graph.get_connected_components()
            z_graph_list.append(z_graph)

    


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
    rg.get_z_graphs(dimensions=[1025, 1025, 21], 
                   correction=[0,0,300], 
                   epsilon=100)
    process_z_graphs("/media/nilsec/d0/gt_mt_data/experiments/z_graphs_200")
