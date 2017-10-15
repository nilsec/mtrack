from evaluation import OptMatch, get_lines, interpolate_nodes
import graphs
from scipy.ndimage.morphology import generate_binary_structure, binary_dilation, distance_transform_edt
from scipy.ndimage.measurements import label
from scipy.spatial import KDTree
import os
from preprocessing import g1_to_nml
import numpy as np
import pdb
import pickle
import h5py
import neuroglancer

def view(raw, seg, voxel_size, offset):
    raw = h5py.File(raw)["data/raw"]
    viewer = neuroglancer.Viewer(voxel_size)
    viewer.add(raw, name="raw")
    viewer.add(seg, name="seg", volume_type="segmentation", offset=offset)
    print viewer

def line_segment_distance(line_p0, line_p1, point):
    line_p0 = np.array(line_p0)
    line_p1 = np.array(line_p1)
    point = np.array(point)

    line_vec = line_p1 - line_p0
    w_0 = point - line_p0
    w_1 = point - line_p1

    #Check if point before, after, or parallel to line segment:
    if np.linalg.norm(line_vec) > 0:
        if np.dot(w_0, line_vec) < 0:
            return np.linalg.norm(point - line_p0)
        elif np.dot(w_1, line_vec) > 0:
            return np.linalg.norm(point - line_p1)
        else:
            d = np.linalg.norm(np.cross(w_0, line_vec))/np.linalg.norm(line_vec)
            return d
    else:
        return None


class Cluster(OptMatch):
    def __init__(self, solution_file, voxel_size=[5.,5.,50], chunk_size=None):
        self.solution_file = solution_file
        self.chunk_size = chunk_size
        
        self.lines = self.__get_lines()
        #self.lines_itp = interpolate_nodes(self.lines, voxel_size=voxel_size)
        if chunk_size is not None:
            self.chunks, self.chunk_positions, self.inv_gt_chunk_positions, _ = OptMatch.get_chunks(self, 
                                                                                                    self.lines_itp, 
                                                                                                    chunk_size)

    def __get_lines(self):
        if self.solution_file[-1] == "/":
            line_base_dir = os.path.dirname(self.solution_file[:-1]) + "/lines"
        else:
            line_base_dir = os.path.dirname(self.solution_file) + "/lines"

        rec_line_dir = line_base_dir + "/reconstruction"
        if not os.path.exists(rec_line_dir):
            line_paths = get_lines(self.solution_file, 
                                   rec_line_dir + "/", 
                                   nml=True)
        else:
            line_paths = [rec_line_dir + "/" + f for f in os.listdir(rec_line_dir) if f.endswith(".gt")]

        return line_paths


    def get_distance_map(self, distance_threshold, voxel_size):
        base_chunk_map = {}
        empty_chunks = []

        for l_base in range(len(self.lines_itp)):
            print "line %s/%s" % (l_base, len(self.lines_itp))
            reached_end_base = False
            c_base = 0
            while not reached_end_base:
                try:
                    chunk = self.chunks[(l_base, c_base)]
                    if chunk:
                        line_p0 = np.array(voxel_size) * np.array(chunk[0])
                        line_p1 = np.array(voxel_size) * np.array(chunk[-1])
                    else:
                        empty_chunks.append((l_base, c_base))
                        c_base += 1
                        continue

                except KeyError:
                    reached_end_base = True
                
                base_chunk_map[(l_base, c_base)] = []
                for l_cmp in range(l_base + 1, len(self.lines_itp)):
                    reached_end_cmp = False
                    c_cmp = 0
                    while not reached_end_cmp:
                        try:
                            chunk_pos_vox = self.chunk_positions[(l_cmp, c_cmp)]
                            chunk_pos_phys = np.array(voxel_size) * chunk_pos_vox

                            d_base_cmp = line_segment_distance(line_p0,
                                                               line_p1,
                                                               chunk_pos_phys)
                            if d_base_cmp is not None:
                                if d_base_cmp < distance_threshold:
                                    base_chunk_map[(l_base, c_base)].append([(l_cmp, c_cmp), d_base_cmp])
                            
                            c_cmp += 1
                        except KeyError:
                            reached_end_cmp = True

                c_base += 1 

        return base_chunk_map

    def get_min_canvas(self, line, dilation, canvas_shape, offset):
        xyz = [None, None, None]
        canvas_shape = np.array(canvas_shape[::-1])
    
        for j in range(3):
            xyz[j] = sorted(line, key=lambda x: x[j])
    
        min_x = xyz[0][0][0]
        min_y = xyz[1][0][1]
        min_z = xyz[2][0][2]

        max_x = xyz[0][-1][0]
        max_y = xyz[1][-1][1]
        max_z = xyz[2][-1][2]

        min_bb = np.array([min_x, min_y, min_z]) - offset
        max_bb = np.array([max_x, max_y, max_z]) - offset

        min_bb -= 2 * dilation
        max_bb += 2 * dilation
        min_bb[min_bb<0] = 0
        max_bb[np.where(max_bb>canvas_shape)] = canvas_shape[np.where(max_bb>canvas_shape)]

        return min_bb, max_bb


    def get_overlap(self, epsilon, min_overlap, offset, voxel_size, canvas_shape, save_volume=None, dt=False):
        canvas_base = np.zeros(canvas_shape)
        dilation = np.array(np.floor(epsilon/np.array(voxel_size)), dtype=int)
        
        # This assumes that the x,y res is the same!
        assert(voxel_size[0] == voxel_size[1])
        struc_all = generate_binary_structure(3, 1)
        
        struc_xy = generate_binary_structure(3, 1)
        struc_xy[0,1,1] = False
        struc_xy[2,1,1] = False

        n_xy = dilation[0] - dilation[2]
        n_z = dilation[2]

        n = 0
        for line in self.lines_itp:
            print "%s/%s" % (n, len(self.lines_itp)) 
            canvas = np.zeros(np.shape(canvas_base))
            bb_min, bb_max = self.get_min_canvas(line, dilation, np.shape(canvas_base), offset)
            
            for voxel in line:
                voxel -= offset
                canvas[voxel[2], voxel[1], voxel[0]] = 1
            
            if not dt:
                canvas = binary_dilation(canvas[bb_min[2]:bb_max[2], bb_min[1]:bb_max[1], bb_min[0]:bb_max[0]], struc_xy, n_xy)
                canvas = binary_dilation(canvas, struc_all, n_z)

            else:
                canvas = distance_transform_edt(canvas[bb_min[2]:bb_max[2], bb_min[1]:bb_max[1], bb_min[0]:bb_max[0]], 
                                            sampling=voxel_size[::-1])

                canvas = (canvas <= epsilon).astype(np.uint8)

            canvas_base[bb_min[2]:bb_max[2], bb_min[1]:bb_max[1], bb_min[0]:bb_max[0]] += canvas
            n += 1

        labeled_volume, n_components = label(canvas_base>=min_overlap)
        
        if save_volume is not None:
            f = h5py.File(save_volume, "w")
            group = f.create_group("data")
            group.create_dataset("overlap", data=labeled_volume.astype(np.dtype(np.uint8)))
 
        return labeled_volume.astype(np.uint8), n_components

    @staticmethod
    def range_intersect(r1, r2):
        return bool(range(max(r1[0], r2[0]), min(r1[-1], r2[-1])+1))

    @staticmethod
    def do_overlap(bb_0_min, bb_0_max, bb_1_min, bb_1_max):
        if not Cluster.range_intersect(range(bb_0_min[2], bb_0_max[2]), range(bb_1_min[2], bb_1_max[2])):
            return False
        elif not Cluster.range_intersect(range(bb_0_min[1], bb_0_max[1]), range(bb_1_min[1], bb_1_max[1])):
            return False
        elif not Cluster.range_intersect(range(bb_0_min[0], bb_0_max[0]), range(bb_1_min[0], bb_1_max[0])):
            return False
        else:
            return True

    def get_line_nodes(self, update_orientations=True, orientation_weighting=None, voxel_size=[5.,5.,50.]):
        reeb_graph = graphs.G1(0)
        line_nodes = []

        print "Get line nodes...\n"
        n = 0
        for line in self.lines:
            print "Line %s\%s" % (n, len(self.lines))
            g1 = graphs.g1_graph.G1(0)
            g1.load(line)

            node_positions = []
            node_orientations = []
            node_pos_oris = []
            reeb_vertices = []
            reeb_line_map = {}
            reeb_line_map_inv = {}
            for v in g1.get_vertex_iterator():
                node_positions.append(np.array(g1.get_position(v)))
                
                u = reeb_graph.add_vertex()
                reeb_graph.set_position(u, node_positions[-1])
                orientation_v = np.array([0.,0.,0.])

                # Update orientation to be the mean from in and out edge
                # constraint to positive quadrant, otherwise ill defined
                if update_orientations:
                    neighbours = g1.get_neighbour_nodes(v)
                    n_neighbours = len(neighbours)

                    for j in range(n_neighbours):
                        vec_abs = np.abs(node_positions[-1] - np.array(g1.get_position(neighbours[j])))
                        # Normalize
                        orientation_v += vec_abs/np.linalg.norm(vec_abs)

                    orientation_v /= float(n_neighbours)
                    if orientation_weighting is not None:
                        orientation_v *= orientation_weighting/np.sqrt(2)
                    
                else:
                    orientation_v = g1.get_orientation(v)
                        
                node_orientations.append(orientation_v)
                node_pos_oris.append(np.hstack((node_positions[-1] * np.array(voxel_size), node_orientations[-1])))
                reeb_graph.set_orientation(u, orientation_v)
                reeb_line_map[u] = v
                reeb_line_map_inv[v] = u
                reeb_vertices.append(u)

            line_nodes.append([node_positions, node_orientations, node_pos_oris, reeb_vertices, reeb_line_map, g1, reeb_line_map_inv])
            n += 1

        return line_nodes, reeb_graph
            
    
    def get_kdtree_groups(self, epsilon, voxel_size, output_dir, orientation_weighting=None):

        line_nodes, reeb_graph = self.get_line_nodes(orientation_weighting=orientation_weighting) 

        #pre_compute trees:
        if orientation_weighting is None:
            trees = [KDTree(line[0] * np.array(voxel_size)) for line in line_nodes]
        else:
            trees = [KDTree(line[2]) for line in line_nodes]
        
        for line_id in range(len(line_nodes)):
            print line_id
            for line_id_cmp in range(line_id + 1, len(line_nodes)):
                hit = trees[line_id].query_ball_tree(trees[line_id_cmp], 2 * epsilon)
                for v_line in range(len(hit)):
                    for v_comp in hit[v_line]:
                        reeb_graph.add_edge(line_nodes[line_id][3][v_line], line_nodes[line_id_cmp][3][v_comp])
        
        if not os.path.exists(os.path.join(output_dir, "reeb_ccs")):
            os.makedirs(os.path.join(output_dir, "reeb_ccs"))

        cc_path_list = reeb_graph.get_components(1, os.path.join(output_dir, "reeb_ccs"))
        reduced_reeb_graph = graphs.G1(0)
        
        cc_vertex_map = {}

        for cc in cc_path_list:
            cc_graph = graphs.G1(0)
            cc_graph.load(cc)
            positions = cc_graph.get_position_array()
            orientations = cc_graph.get_orientation_array()
            mean_position = np.rint(np.mean(positions, axis=1))
            mean_orientation = np.mean(orientations, axis=1)
            v = reduced_reeb_graph.add_vertex()
            cc_vertex_map[v] = [u for u in cc_graph.get_vertex_iterator()]
            reduced_reeb_graph.set_position(v, mean_position)
            reduced_reeb_graph.set_orientation(v, mean_orientation)

        
        i = 0
        for line in line_nodes:
            print "Connect line %s" % i
            i += 1
            reeb_line_map = line[4]
            reeb_line_map_inv = line[6]
            g1_line = line[5]
            reeb_vertices = line[3]
            
            for reeb_node in reeb_vertices:
                line_node = reeb_line_map[reeb_node]
                line_neighbours = g1_line.get_neighbour_nodes(line_node)

                for v in line_neighbours:
                    reeb_node_neighbour = reeb_line_map_inv[v]
                    for reduced_v, reeb_list_v in cc_vertex_map.iteritems():
                        if reeb_node in reeb_list_v:
                            reduced_base_node = reduced_v
                        if reeb_node_neighbour in reeb_list_v:
                            reduced_partner_node = reduced_v

                    reduced_reeb_graph.add_edge(reduced_base_node, reduced_partner_node)
                            
        g1_to_nml(reeb_graph, os.path.join(output_dir, "reeb_%s_%s.nml") % (orientation_weighting, epsilon), 
                                           knossify=True, 
                                           voxel_size=voxel_size)
        
        g1_to_nml(reduced_reeb_graph, os.path.join(output_dir, "reduced_reeb_%s_%s.nml") % (orientation_weighting, epsilon), 
                                           knossify=True, 
                                           voxel_size=voxel_size)
            

    def get_groups(self, epsilon, offset, voxel_size, canvas_shape):
        dilation = np.array(np.floor(epsilon/np.array(voxel_size)), dtype=int)
        
        # This assumes that the x,y res is the same!
        assert(voxel_size[0] == voxel_size[1])
        struc_all = generate_binary_structure(3, 1)
        
        struc_xy = generate_binary_structure(3, 1)
        struc_xy[0,1,1] = False
        struc_xy[2,1,1] = False

        n_xy = dilation[0] - dilation[2]
        n_z = dilation[2]
        
        canvas_dict = {}

        line_id = 0
        for line in self.lines_itp:
            print "%s/%s" % (line_id, len(self.lines_itp)) 
            bb_min, bb_max = self.get_min_canvas(line, dilation, canvas_shape, offset)
            try:
                canvas = canvas_dict[line_id]
            except KeyError:
                canvas = np.zeros(canvas_shape)

                for voxel in line:
                    v = voxel - offset
                    canvas[v[2], v[1], v[0]] = 1

                canvas = binary_dilation(canvas[bb_min[2]:bb_max[2], 
                                                    bb_min[1]:bb_max[1], 
                                                    bb_min[0]:bb_max[0]], 
                                                    struc_xy, n_xy)
                        
                canvas = binary_dilation(canvas, struc_all, n_z)
                canvas_dict[line_id] = canvas
 
            line_id_cmp = line_id + 1
            for line_cmp in self.lines_itp[line_id + 1:]:
                bb_min_cmp, bb_max_cmp = self.get_min_canvas(line_cmp, dilation, canvas_shape, offset)
                
                # check if bb's overlap, if not skip canvas creation
                if not Cluster.do_overlap(bb_min, bb_max, bb_min_cmp, bb_max_cmp):
                    line_id_cmp += 1
                    continue
                else:
                    canvas_both = np.zeros(canvas_shape)
                    
                    try:
                        canvas_cmp = canvas_dict[line_id_cmp]
                    except KeyError:
                        canvas_cmp = np.zeros(canvas_shape)
                    
                        for voxel_cmp in line_cmp:
                            v_cmp = voxel_cmp - offset
                            canvas_cmp[v_cmp[2], v_cmp[1], v_cmp[0]] = 1

                        canvas_cmp = binary_dilation(canvas_cmp[bb_min_cmp[2]:bb_max_cmp[2], 
                                                                bb_min_cmp[1]:bb_max_cmp[1], 
                                                                bb_min_cmp[0]:bb_max_cmp[0]], 
                                                                struc_xy, n_xy)

                        canvas_cmp = binary_dilation(canvas_cmp, struc_all, n_z)
                        canvas_dict[line_id_cmp] = canvas_cmp
                    
                    canvas_both[bb_min[2]:bb_max[2], bb_min[1]:bb_max[1], bb_min[0]:bb_max[0]] += canvas
                    canvas_both[bb_min_cmp[2]:bb_max_cmp[2], bb_min_cmp[1]:bb_max_cmp[1], bb_min_cmp[0]:bb_max_cmp[0]] += canvas_cmp
                    print line_id, line_id_cmp, "\n", np.where(canvas_both == 2), "\n\n"

                    line_id_cmp += 1

            line_id += 1
 
        return 0
 


if __name__ == "__main__":
    test_solution = "/media/nilsec/d0/gt_mt_data/solve_volumes/test_volume_grid32_ps035035_300_399/solution/volume.gt" 
    validation_solution = "/media/nilsec/d0/gt_mt_data/solve_volumes/grid_2/grid_32/solution/volume.gt"
    """
    cluster = Cluster(test_solution)
    labeled_volume, n_comps = cluster.get_overlap(epsilon=75,
                                                  min_overlap=2, 
                                                  offset=np.array([0,0,300]),
                                                  voxel_size=[5.,5.,50.],
                                                  canvas_shape=[100,1024,1024],
                                                  save_volume="/media/nilsec/d0/gt_mt_data/data/Test/cluster.h5")

    raw_test = "/media/nilsec/d0/gt_mt_data/data/Test/raw_split.h5"
    raw_validation = "/media/nilsec/d0/gt_mt_data/data/Validation/raw.h5"
    view(raw_test, labeled_volume, voxel_size=[5.,5.,50.], offset=[0,0,300*50])
    """
    cluster = Cluster(test_solution)
    cluster.get_kdtree_groups(epsilon=100, 
                              output_dir="/media/nilsec/d0/gt_mt_data/experiments/clustering/v1",
                              orientation_weighting=float(500),
                              voxel_size=[5.,5.,50.])


