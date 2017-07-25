import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import graphs
from graphs.graph import G
import evaluation
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pdb
import itertools
import os

def shortest_path_eval(gt_line_dir,
                       rec_line_dir,
                       dimensions,
                       correction,
                       tolerance_radius,
                       voxel_size,
                       min_path_length,
                       plot_sp=False,
                       plot_error_graph=True):

    gt_lines = [os.path.join(gt_line_dir, f) for f in os.listdir(gt_line_dir) if f.endswith(".gt")]
    rec_lines = [os.path.join(rec_line_dir, f) for f in os.listdir(rec_line_dir) if f.endswith(".gt")]
    
    rec_chains, gt_volume = get_rec_chains(rec_lines,
                                           gt_lines,
                                           dimensions,
                                           tolerance_radius,
                                           voxel_size,
                                           correction=correction)

    if plot_sp:
        for chain in rec_chains:
            chain.get_shortest_path(True)

    error_graph = ErrorGraph(rec_chains, gt_volume)
    error_graph.get_sp_matching(min_path_length)
    
    if plot_error_graph:
        error_graph.draw()
    
    return error_graph.get_report()


class GtVolume(object):
    def __init__(self):
        self.id_dict = {}
        self.ids = set()

    def add_volume(self, volume, id):
        self.ids.add(id)
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

class RecChain(graphs.graph.G):
    def __init__(self):
        G.__init__(self, 0, None)
        G.new_edge_property(self, "weight", dtype="short")
        G.new_vertex_property(self, "id", dtype="int")
        G.new_vertex_property(self, "layer", dtype="long")
        
        self.layers = 0
        self.previous_layer = []
        self.structure = {}
        self.ids = set()

        #Dummy Start Node
        self.start_dummy = None
        self.add_voxel([-1])
        
        #Dummy end node
        self.end_dummy = None

        #Shortest Path ID List
        self.sp_ids = []

    def get_id(self, u):
        return G.get_vertex_property(self, "id")[u]

    def add_voxel(self, ids):
        assert(len(ids)>0)
        [self.ids.add(id) for id in ids]
        current_layer = []

        for id in ids:
            v = G.add_vertex(self)
            
            if id == -1:
                self.start_dummy = v
            if id == -2:
                self.end_dummy = v

            G.set_vertex_property(self, "id", v, id)
            G.set_vertex_property(self, "layer", v, self.layers)
                
            if self.previous_layer:
                # Connect to all nodes of previous layer
                for u in self.previous_layer:
                    e = G.add_edge(self, v, u)
                    
                    # Add edge weight
                    id_u = G.get_vertex_property(self, "id")[u]
                    if id_u == id:
                        G.set_edge_property(self, "weight", e.source(), e.target(), 0)
                    else:
                        G.set_edge_property(self, "weight", e.source(), e.target(), 1)

            current_layer.append(v)
                
        self.structure[self.layers] = current_layer
        self.previous_layer = current_layer
        self.layers += 1

    def draw(self, show_edge_weights=True, 
                   show_ids=True, 
                   vertex_highlight=[], 
                   edge_highlight=[]):

        color_wheel = cm.rainbow(np.linspace(0, 1, max([max(self.ids), 4]) + 1))
        vertex_pos = {}

        plt.figure()
        for layer_id, layer_vertices in self.structure.iteritems():
            xx = range(len(layer_vertices))
            y = self.layers - layer_id

            for x in xx:
                vertex_id = G.get_vertex_property(self, "id")[layer_vertices[x]]

                if layer_vertices[x] in vertex_highlight:
                    marker = "D"
                else:
                    marker = "o"

                plt.scatter(x, y, c=color_wheel[vertex_id - 1], marker=marker)
                vertex_pos[layer_vertices[x]] = (x,y)
                if show_ids:
                    plt.annotate(str(vertex_id), (x,y))

        for edge in G.get_edge_iterator(self):
            start = vertex_pos[edge.source()]
            end = vertex_pos[edge.target()]
            line = zip(start, end)
            if edge in edge_highlight:
                linestyle = "-"
                linewidth = 2
            else:
                linestyle = "--"
                linewidth = 1
            plt.plot(line[0], line[1], c='black', linestyle=linestyle, linewidth=linewidth)
            
            if show_edge_weights:
                weight = G.get_edge_property(self, "weight", edge.source(), edge.target())
                x = (start[0] + end[0])/2.0
                y = (start[1] + end[1])/2.0
                plt.annotate(str(weight), (x,y))
    
        plt.show()

    def get_sp_ids(self):
        return self.sp_ids

    def get_shortest_path(self, plot=False):
        if self.end_dummy is None:
            self.add_voxel([-2])
            
        vertex_list, edge_list = G.get_shortest_path(self, 
                                                     self.start_dummy, 
                                                     self.end_dummy,
                                                     G.get_edge_property(self, "weight"))

        for v in vertex_list:
            self.sp_ids.append(self.get_id(v))

        if plot:
            self.draw(vertex_highlight=vertex_list,
                      edge_highlight=edge_list)

        return vertex_list, edge_list

class ErrorGraph(graphs.graph.G):
    def __init__(self, rec_chain_list, gt_volume):
        self.gt_ids = gt_volume.ids
        self.rec_chain_list = rec_chain_list
        self.min_id_sets = []
 
        G.__init__(self, 0, None)
        G.new_vertex_property(self, "id", dtype="int")

        self.rec_layer = []
        self.gt_layer = []
        
    def get_min_id_groups(self, id_chain, min_length):
        min_id_groups = []
        
        for id, group in itertools.groupby(id_chain):
            if len(list(group)) >= min_length:
                if id != (-1) and id != (-2):
                    min_id_groups.append(id)

        return min_id_groups
        
    def get_sp_matching(self, min_id_path=1):
        N_rec = 0

        rec_gt = {}
        for chain in self.rec_chain_list:
            chain.get_shortest_path()
            id_chain = chain.get_sp_ids()
            assert(bool(id_chain)) # No empty chains

            min_id_chain = self.get_min_id_groups(id_chain, min_length=min_id_path)
            self.min_id_sets.append(set(min_id_chain))

            rec_node = G.add_vertex(self)
            G.set_vertex_property(self, "id", rec_node, N_rec)
            self.rec_layer.append(rec_node)
            rec_gt[rec_node] = tuple(set(min_id_chain))
            
            N_rec += 1
            
        for id in self.gt_ids:
            gt_node = G.add_vertex(self)
            G.set_vertex_property(self, "id", gt_node, id)
            self.gt_layer.append(gt_node)

        for rec_node, gt_ids in rec_gt.iteritems():
            for gt_node in self.gt_layer:
                if G.get_vertex_property(self, "id")[gt_node] in gt_ids:
                    G.add_edge(self, rec_node, gt_node)



    def draw(self):
        plt.figure()
        
        x = 0
        y_rec = 1
        vertex_pos = {}
        for rec_node in self.rec_layer:
            plt.scatter(x, y_rec)
            vertex_pos[rec_node] = (x, y_rec)
            vertex_id = G.get_vertex_property(self, "id")[rec_node]
            plt.annotate(str(vertex_id), (x, y_rec))
            x += 1

        x = 0
        y_gt = 4
        for gt_node in self.gt_layer:
            plt.scatter(x, y_gt)
            vertex_pos[gt_node] = (x, y_gt)
            vertex_id = G.get_vertex_property(self, "id")[gt_node]
            plt.annotate(str(vertex_id), (x,y_gt))
            x += 1

        for edge in G.get_edge_iterator(self):
            start = vertex_pos[edge.source()]
            end = vertex_pos[edge.target()]
            line = zip(start, end)
            plt.plot(line[0], line[1], c='black')

        plt.show()

    def get_report(self, remove_background=True):
        print "Get report..."
        print "Remove background: ", str(remove_background)
        rec_neighbours = {}
        gt_neighbours = {}
        false_positive = 0
        false_negative = 0
        mergers = 0
        splits = 0

        for rec_vertex in self.rec_layer:
            matches = G.get_neighbour_nodes(self, rec_vertex)

            if remove_background:
                for match in matches:
                    if G.get_vertex_property(self, "id")[match] == 0:
                        matches.remove(match)
            
            n_matches = len(matches)

            if n_matches == 0:
                false_positive += 1
                continue

            mergers += (n_matches - 1)

        for gt_vertex in self.gt_layer:
            matches = G.get_neighbour_nodes(self, gt_vertex)
            n_matches = len(matches)

            if n_matches == 0:
                false_negative += 1
                continue
        
            splits += (n_matches - 1)

        
        return {"fp": false_positive, 
                "fn": false_negative, 
                "mergers": mergers, 
                "splits": splits}
        
def get_rec_chains(rec_line_list, 
                   gt_line_list, 
                   dimensions, 
                   tolerance_radius,
                   voxel_size,
                   correction=np.array([0,0,0])):

    gt_volume = get_gt_volume(gt_line_list,
                              dimensions,
                              tolerance_radius,
                              voxel_size,
                              correction)

    rec_chains = []
    for l in rec_line_list:
        voxel_line = get_rec_line(l)
        rec_chain = RecChain()

        for voxel_pos in voxel_line:
            voxel_ids = gt_volume.get_ids(tuple(voxel_pos))
            rec_chain.add_voxel(voxel_ids)
        
        rec_chains.append(rec_chain)

    return rec_chains, gt_volume
        

def get_rec_line(rec_line, correction=np.array([0,0,0])):
    if isinstance(rec_line, str):
        g1 = graphs.g1_graph.G1(0)
        g1.load(rec_line)

    elif isinstance(rec_line, graphs.g1_graph.G1):
        g1 = rec_line

    else:
        raise TypeError("A line must be a path or a g1 graph.")
 
    line_points = []
        
    for edge in g1.get_edge_iterator():
        start = np.array(g1.get_position(edge.source()), dtype=int)
        start -= correction
        end = np.array(g1.get_position(edge.target()), dtype=int)
        end -= correction

        dda = evaluation.DDA3(start, end)
        skeleton_edge = dda.draw()
        skeleton_edge = [np.array([p[2], p[1], p[0]]) for p in skeleton_edge]
        line_points.extend(skeleton_edge)
        
    return line_points
 

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