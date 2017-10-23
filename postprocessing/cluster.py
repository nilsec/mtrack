from evaluation import OptMatch, get_lines, interpolate_nodes
import graphs
from scipy.ndimage.morphology import generate_binary_structure, binary_dilation, distance_transform_edt
from scipy.ndimage.measurements import label
from scipy.spatial import KDTree
import os
from preprocessing import g1_to_nml
from combine_solutions import combine_knossos_solutions
import numpy as np
import pdb
import pickle
import h5py
import neuroglancer


class VCluster(object):
    def __init__(self, solution_file, voxel_size=[5.,5.,50], lines=None):
        self.solution_file = solution_file
        self.voxel_size = voxel_size
        if lines is None:
            self.lines = get_lines_from_file(solution_file)
            self.lines_itp = interpolate_nodes(self.lines, voxel_size=voxel_size)
        else:
            self.lines_itp = lines
    
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

    def __process_lines(self, update_ori=True, weight_ori=None, lines=None):
        reeb_graph = graphs.G1(0)
        reeb_graph.new_vertex_property("line_neighbours", dtype="vector<int>")
        reeb_graph.new_vertex_property("line_vertex", dtype="int")

        processed_lines = []

        print "Process lines..."
        n = 0
        v0 = 0

        if lines is None:
            lines = self.lines

        for line in lines:
            if isinstance(line, str):
                l_graph = graphs.g1_graph.G1(0)
                l_graph.load(line)
            else:
                l_graph = line
            

            line_attr = {"pos": [],
                         "ori": [],
                         "pos_ori": [],
                         "v_reeb": [],
                         "v_line": [],
                         "l_graph": l_graph}

            n_line_vertices = 0
            v0 += n_line_vertices
            for v in l_graph.get_vertex_iterator():
                pos = np.array(l_graph.get_position(v)) * np.array(self.voxel_size)

                ori = np.array([0.,0.,0.])
                neighb = l_graph.get_neighbour_nodes(v)
                
                if update_ori:
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
                reeb_graph.set_vertex_property("line_neighbours", v_reeb, np.array(neighb, dtype=int) + v0)
                reeb_graph.set_vertex_property("line_vertex", v_reeb, int(v) + v0)

                reeb_graph.set_position(v_reeb, pos)
                reeb_graph.set_orientation(v_reeb, ori)
                
                line_attr["pos"].append(pos)
                line_attr["ori"].append(ori)
                line_attr["pos_ori"].append(pos_ori)
                line_attr["v_reeb"].append(v_reeb)
                line_attr["v_line"].append(int(v) + v0)
                
                n_line_vertices += 1

            processed_lines.append(line_attr)
            n += 1

        return processed_lines, reeb_graph
            
    
    def cluster_vertices(self, epsilon, output_dir, use_ori=True, weight_ori=None, lines=None):

        processed_lines, reeb_graph = self.__process_lines(update_ori=use_ori, 
                                                           weight_ori=weight_ori,
                                                           lines=lines) 

        print "Initialize trees..." 
        if use_ori:
            trees = [KDTree(line_attr["pos_ori"]) for line_attr in processed_lines]
        else:
            trees = [KDTree(line_attr["pos"]) for line_attr in processed_lines]

        
        print "Query ball trees..."        
        for line_id in range(len(processed_lines)):
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

        return reeb_graph


    def cluster_lines(self, 
                      epsilon_lines, 
                      p_hits, 
                      min_lines,
                      remove_aps,
                      min_k,
                      sbm,
                      nested,
                      edge_weights,
                      output_dir, 
                      use_ori=True, 
                      weight_ori=None,
                      epsilon_vertices=None,
                      v_cluster=None):
        
        processed_lines, _ = self.__process_lines(update_ori=use_ori, 
                                                           weight_ori=weight_ori) 

        print "Initialize trees..." 
        if use_ori:
            trees = [KDTree(line_attr["pos_ori"]) for line_attr in processed_lines]
        else:
            trees = [KDTree(line_attr["pos"]) for line_attr in processed_lines]

        line_graph = graphs.G1(0)
        for v_id in range(len(processed_lines)):
            line_graph.add_vertex()

        edge_weight = line_graph.new_edge_property("weight", "float")
        print "Query ball trees..."        
        weights = []
        for line_id in range(len(processed_lines)):
            print line_id
            for line_id_cmp in range(line_id + 1, len(processed_lines)):
                #print "cmp %s" % line_id_cmp
                hits = trees[line_id].query_ball_tree(trees[line_id_cmp], 2 * epsilon_lines)
                n_hits = 0

                for hit in hits:
                    if hit:
                        n_hits += 1

                if n_hits>=1 and edge_weights:
                    line_graph.add_edge(line_id, line_id_cmp)
                    weights.append(n_hits/float(len(hits)))                    
                    #line_graph.set_edge_property("weight", line_id, line_id_cmp, n_hits/float(len(hits)), e)
                    #assert(sbm)
                elif p_hits is not None:
                    if n_hits/float(len(hits)) >= p_hits:
                        line_graph.add_edge(line_id, line_id_cmp)
 

        edge_weight = line_graph.new_edge_property("weight", "float", vals=weights)
 
        cc_path_list = line_graph.get_components(min_lines, 
                                                 os.path.join(output_dir, "line_ccs/"), 
                                                 remove_aps, 
                                                 min_k, 
                                                 sbm,
                                                 nested,
                                                 edge_weights)

        if not cc_path_list:
            raise Warning("No matching lines")

        print "Combine Lines.."
        n = 0
        for cc in cc_path_list:
            cc_graph = graphs.G1(0)
            cc_graph.load(cc)
            
            line_cluster = graphs.G1(0)
            
            line_cluster_dir = os.path.join(output_dir, "line_cluster/c%s" % n) 
            if not os.path.exists(line_cluster_dir):
                os.makedirs(line_cluster_dir)
            
            l_graph_cluster = []
            for v in cc_graph.get_vertex_iterator():
                l_graph = processed_lines[int(v)]["l_graph"]
                g1_to_nml(l_graph,
                          line_cluster_dir + "/line_%s.nml" % int(v),
                          knossify=True,
                          voxel_size=self.voxel_size)
                l_graph_cluster.append(l_graph)

            if epsilon_vertices is not None:
                self.cluster(epsilon_vertices, output_dir + "/lv_cluster", lines=l_graph_cluster, weight_ori=weight_ori)

            combine_knossos_solutions(line_cluster_dir, os.path.join(output_dir, "combined/combined_%s.nml" % n), tag="line")
            n += 1            

        return 0

 
    def reduce_cluster(self, reeb_graph, output_dir, min_vertices, remove_aps, min_k, sbm):
        print "Reduce cluster..."
        cc_path_list = reeb_graph.get_components(min_vertices, os.path.join(output_dir, "reeb_ccs/"), remove_aps, min_k, sbm)
        
        reduced_reeb_graph = graphs.G1(0)
        reduced_reeb_graph.new_vertex_property("line_neighbours", "vector<int>")
        reduced_reeb_graph.new_vertex_property("line_vertices", "vector<int>")

        for cc in cc_path_list:
            cc_graph = graphs.G1(0)
            cc_graph.load(cc)

            g1_to_nml(cc_graph, 
                      os.path.join(output_dir, "reeb_ccs/nml/%s" % os.path.basename(cc).replace(".gt", ".nml")), 
                      knossos=True, 
                      voxel_size=self.voxel_size)
            
            cc_positions = cc_graph.get_position_array()
            cc_orientations = cc_graph.get_orientation_array()
            
            cc_mean_position = np.mean(cc_positions, axis=1)
            cc_mean_orientation = np.mean(cc_orientations, axis=1)

            v_reduced = reduced_reeb_graph.add_vertex()
            reduced_reeb_graph.set_position(v_reduced, cc_mean_position)
            reduced_reeb_graph.set_orientation(v_reduced, cc_mean_orientation)

            cc_line_vertices = []
            cc_line_neighbours = set()
           
            for v in cc_graph.get_vertex_iterator():
                [cc_line_neighbours.add(u) for u in cc_graph.get_vertex_property("line_neighbours", v)]
                cc_line_vertices.append(cc_graph.get_vertex_property("line_vertex", v))
            
            reduced_reeb_graph.set_vertex_property("line_neighbours", v_reduced, np.array([u for u in cc_line_neighbours], dtype=int))
            reduced_reeb_graph.set_vertex_property("line_vertices", v_reduced, np.array(cc_line_vertices, dtype=int))

        return reduced_reeb_graph

    def connect_cluster(self, reduced_reeb_graph, output_dir):
        print "Connect reduced cluster..."
        for v in reduced_reeb_graph.get_vertex_iterator():
            v_vertices = reduced_reeb_graph.get_vertex_property("line_vertices", v)
            for u in range(int(v) + 1, reduced_reeb_graph.get_number_of_vertices()):
                u_neighbours = reduced_reeb_graph.get_vertex_property("line_neighbours", u)
                intersect = np.intersect1d(v_vertices, u_neighbours)
                if intersect.size:
                    reduced_reeb_graph.add_edge(u,v)

        g1_to_nml(reduced_reeb_graph, 
                  os.path.join(output_dir, "reduced_reeb.nml"), 
                  knossos=True, 
                  voxel_size=self.voxel_size)

        return reduced_reeb_graph


    def cluster(self, epsilon, output_dir, use_ori=True, weight_ori=None, min_vertices=1, remove_aps=False, min_k=1, sbm=False, lines=None):
        reeb_graph = self.cluster_vertices(epsilon, output_dir, use_ori, weight_ori, lines=lines)
        reduced_graph = self.reduce_cluster(reeb_graph, output_dir, min_vertices=min_vertices, remove_aps=remove_aps, min_k=min_k, sbm=sbm)
        connected_graph = self.connect_cluster(reduced_graph, output_dir)


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
    sc_vertices = False
    sc_lines=True
   
    
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
    if sc_vertices:
        scluster = SCluster(test_solution, voxel_size=[5.,5.,50.])
        scluster.cluster(epsilon=50, 
                         output_dir="/media/nilsec/d0/gt_mt_data/experiments/clustering/v3",
                         use_ori=True,
                         weight_ori=float(700),
                         min_vertices=3,
                         remove_aps=True,
                         sbm=False,
                         min_k=1)

    if sc_lines:
        p_hits=None
        epsilon_lines=100
        min_lines=1
        weight_ori = 700
        epsilon_vertices = 50

        scluster = SCluster(test_solution, voxel_size=[5.,5.,50.])
        scluster.cluster_lines(epsilon_lines=epsilon_lines, 
                               p_hits=p_hits, 
                               min_lines=min_lines,
                               remove_aps=False,
                               min_k=1,
                               sbm=True,
                               nested=True,
                               edge_weights=True,
                               output_dir="/media/nilsec/d0/gt_mt_data/experiments/clustering/el{}_ev{}_min{}_ph{}_wo{}_waps_sbm".format(epsilon_lines, epsilon_vertices, min_lines, p_hits, weight_ori), 
                               use_ori=True, 
                               weight_ori=float(weight_ori),
                               epsilon_vertices=epsilon_vertices)
