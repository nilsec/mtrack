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


class VCluster(object):
    def __init__(self, solution_file, voxel_size=[5.,5.,50]):
        self.solution_file = solution_file
        self.voxel_size = voxel_size
        self.lines = get_lines_from_file(solution_file)
        self.lines_itp = interpolate_nodes(self.lines, voxel_size=voxel_size)
    
    @staticmethod
    def view(raw, seg, voxel_size, offset):
        raw = h5py.File(raw)["data/raw"]
        viewer = neuroglancer.Viewer(voxel_size)
        viewer.add(raw, name="raw")
        viewer.add(seg, name="seg", volume_type="segmentation", offset=offset)
        print viewer
 

    def cluster(self, epsilon, min_overlap, offset, canvas_shape, save_volume=None, dt=False):
        canvas_base = np.zeros(canvas_shape)
        dilation = np.array(np.floor(epsilon/np.array(self.voxel_size)), dtype=int)
        
        # This assumes that the x,y res is the same!
        assert(self.voxel_size[0] == self.voxel_size[1])
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
                                            sampling=self.voxel_size[::-1])

                canvas = (canvas <= epsilon).astype(np.uint8)

            canvas_base[bb_min[2]:bb_max[2], bb_min[1]:bb_max[1], bb_min[0]:bb_max[0]] += canvas
            n += 1

        labeled_volume, n_components = label(canvas_base>=min_overlap)
        
        if save_volume is not None:
            f = h5py.File(save_volume, "w")
            group = f.create_group("data")
            group.create_dataset("overlap", data=labeled_volume.astype(np.dtype(np.uint8)))
 
        return labeled_volume.astype(np.uint8), n_components

    
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


class SCluster(object):
    def __init__(self, solution_file, voxel_size=[5.,5.,50]):
        self.solution_file = solution_file
        self.voxel_size = voxel_size
        self.lines = get_lines_from_file(solution_file)

    def __process_lines(self, update_ori=True, weight_ori=None):
        reeb_graph = graphs.G1(0)
        processed_lines = []

        print "Process lines...\n"
        n = 0

        for line in self.lines:
            print "Line %s\%s" % (n, len(self.lines))
            l_graph = graphs.g1_graph.G1(0)
            l_graph.load(line)

            line_attr = {"pos": [],
                         "ori": [],
                         "pos_ori": [],
                         "v_reeb": [],
                         "v_line": []}

            for v in l_graph.get_vertex_iterator():
                pos = np.array(l_graph.get_position(v)) * np.array(self.voxel_size)

                ori = np.array([0.,0.,0.])
                if update_ori:
                    neighb = l_graph.get_neighbour_nodes(v)
                    
                    for i in range(len(neighb)):
                        vec_abs = np.abs(pos - np.array(l_graph.get_position(neighb[i])))
                        ori += vec_abs/np.linalg.norm(vec_abs)
                    ori /= float(len(neighb))
                    
                    if weight_ori is not None:
                        ori *= weight_ori/np.sqrt(2)
                else:
                    ori = l_graph.get_orientation(v)
 
                pos_ori = np.hstack((pos, ori))
                
                v_reeb = reeb_graph.add_vertex()
                reeb_graph.set_position(v_reeb, pos)
                reeb_graph.set_orientation(v_reeb, ori)
                
                line_attr["pos"].append(pos)
                line_attr["ori"].append(ori)
                line_attr["pos_ori"].append(pos_ori)
                line_attr["v_reeb"].append(v_reeb)
                line_attr["v_line"].append(v)

            processed_lines.append(line_attr)
            n += 1

        return processed_lines, reeb_graph
            
    
    def cluster_vertices(self, epsilon, output_dir, use_ori=True, weight_ori=None):

        processed_lines, reeb_graph = self.__process_lines(update_ori=use_ori, 
                                                           weight_ori=weight_ori) 

        print "Initialize trees...\n" 
        if use_ori:
            trees = [KDTree(line_attr["pos_ori"]) for line_attr in processed_lines]
        else:
            trees = [KDTree(line_attr["pos"]) for line_attr in processed_lines]

        
        print "Query ball trees...\n"        
        for line_id in range(len(processed_lines)):
            print line_id

            for line_id_cmp in range(line_id + 1, len(processed_lines)):
                hit = trees[line_id].query_ball_tree(trees[line_id_cmp], 2 * epsilon)

                for v_line in range(len(hit)):
                    for v_comp in hit[v_line]:
                        reeb_graph.add_edge(processed_lines[line_id]["v_reeb"][v_line], 
                                            processed_lines[line_id_cmp]["v_reeb"][v_comp])
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        g1_to_nml(reeb_graph, 
                  os.path.join(output_dir, "reeb_%s_%s.nml") % (int(weight_ori), int(epsilon)), 
                  knossos=True, 
                  voxel_size=self.voxel_size)

        """
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
        
        g1_to_nml(reduced_reeb_graph, os.path.join(output_dir, "reduced_reeb_%s_%s.nml") % (orientation_weighting, epsilon), 
                                           knossify=True, 
                                           voxel_size=voxel_size)
        """ 

def get_lines_from_file(solution_file):
    if solution_file[-1] == "/":
        line_base_dir = os.path.dirname(solution_file[:-1]) + "/lines"
    else:
        line_base_dir = os.path.dirname(solution_file) + "/lines"

    rec_line_dir = line_base_dir + "/reconstruction"
    if not os.path.exists(rec_line_dir):
        line_paths = get_lines(solution_file, 
                               rec_line_dir + "/", 
                               nml=True)
    else:
        line_paths = [rec_line_dir + "/" + f for f in os.listdir(rec_line_dir) if f.endswith(".gt")]

    return line_paths

    
if __name__ == "__main__":
    test_solution = "/media/nilsec/d0/gt_mt_data/solve_volumes/test_volume_grid32_ps035035_300_399/solution/volume.gt" 
    validation_solution = "/media/nilsec/d0/gt_mt_data/solve_volumes/grid_2/grid_32/solution/volume.gt"
    vc = False
    sc = True
   
    
    #Volume Cluster 
    if vc:
        vcluster = VCluster(test_solution, voxel_size=[5.,5.,50.])
        labeled_volume, n_comps = vcluster.cluster(epsilon=75,
                                                   min_overlap=2, 
                                                   offset=np.array([0,0,300]),
                                                   canvas_shape=[100,1024,1024],
                                                   save_volume="/media/nilsec/d0/gt_mt_data/data/Test/vcluster.h5")

        raw_test = "/media/nilsec/d0/gt_mt_data/data/Test/raw_split.h5"
        raw_validation = "/media/nilsec/d0/gt_mt_data/data/Validation/raw.h5"
        VCluster.view(raw_test, labeled_volume, voxel_size=[5.,5.,50.], offset=[0,0,300*50])

    #Skeleton Cluster
    if sc:
        scluster = SCluster(test_solution, voxel_size=[5.,5.,50.])
        scluster.cluster_vertices(epsilon=100, 
                                  output_dir="/media/nilsec/d0/gt_mt_data/experiments/clustering/v1",
                                  use_ori=True,
                                  weight_ori=float(500))


