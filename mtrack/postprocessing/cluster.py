import os
import numpy as np
import h5py
from scipy.ndimage.morphology import generate_binary_structure, binary_dilation, distance_transform_edt
from scipy.ndimage.measurements import label
from scipy.spatial import KDTree

try:
    import skeletopyze
except ImportError("Skeletopyze module not installed, clustering not available"):
    pass

import mtrack.graphs
from mtrack.evaluation import get_lines, interpolate_nodes
from mtrack.preprocessing import g1_to_nml
from mtrack.postprocessing.combine_solutions import combine_knossos_solutions


def skeletonize(solution_file, 
                output_dir,
                epsilon_lines,
                epsilon_volumes,
                min_overlap_volumes,
                canvas_shape,
                offset,
                orientation_factor,
                remove_singletons,
                use_ori=True,
                voxel_size=[5.,5.,50.]):

    scluster = SCluster(solution_file, voxel_size=[5.,5.,50.])
    cluster = scluster.get_high_connectivity_cluster(epsilon_lines, 
                                                     use_ori=True, 
                                                     weight_ori=orientation_factor,
                                                     remove_singletons=remove_singletons, 
                                                     output_dir=output_dir + "/lines")
    j = 0
    skeletons = []
    for lines in cluster:
        vcluster = VCluster(0, lines=lines)
        volume, n_comps = vcluster.cluster(epsilon=epsilon_volumes,
                                           min_overlap=min_overlap_volumes,
                                           offset=offset,
                                           canvas_shape=canvas_shape,
                                           save_volume=output_dir + "/volumes/v_{}.h5".format(j))

        skeleton = vcluster.skeletonize_volume(volume, 
                                               output_dir + "/skeletons/s_{}".format(j), 
                                               offset, 
                                               voxel_size=voxel_size)
        skeletons.append(skeleton)
        j += 1

    return skeletons


class VCluster(object):
    def __init__(self, solution_file, voxel_size=[5.,5.,50], lines=None, lines_itp=None):
        self.solution_file = solution_file
        self.voxel_size = voxel_size
        if lines is None and lines_itp is None:
            self.lines = get_lines_from_file(solution_file)
            self.lines_itp = interpolate_nodes(self.lines, voxel_size=voxel_size)
        elif lines is not None and lines_itp is None:
            self.lines = lines
            self.lines_itp = interpolate_nodes(self.lines, voxel_size=voxel_size)
        else:
            self.lines_itp = lines_itp
        
    
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
                print voxel
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
            if not os.path.exists(os.path.dirname(save_volume)):
                os.makedirs(os.path.dirname(save_volume))
            f = h5py.File(save_volume, "w")
            group = f.create_group("data")
            group.create_dataset("overlap", data=labeled_volume.astype(np.dtype(np.uint8)))
 
        return labeled_volume.astype(np.uint8), n_components

    def skeletonize_volume(self, volume, output_path, offset, voxel_size=[5.,5.,50.]):
        if isinstance(volume, str):
            f = h5py.File(volume)
            volume = f["data/overlap"].value
        else:
            assert(isinstance(volume, np.ndarray))

        params = skeletopyze.Parameters()
        res = skeletopyze.point_f3()
        res.__setitem__(0,voxel_size[0])
        res.__setitem__(1,voxel_size[1])
        res.__setitem__(2,voxel_size[2])

        b = skeletopyze.get_skeleton_graph(volume, params, res)

        skeleton = mtrack.graphs.G1(0)
        for n in b.nodes():
            v = skeleton.add_vertex()
            skeleton.set_position(v, np.array([b.locations(n).x() + offset[0], 
                                               b.locations(n).y() + offset[1], 
                                               b.locations(n).z() + offset[2]])) 
        for e in b.edges():
            skeleton.add_edge(e.u, e.v)
        
        if output_path is not None:
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            skeleton.save(output_path + ".gt")
            g1_to_nml(skeleton, output_path + ".nml", knossify=True)

        return skeleton       

    
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
        reeb_graph = mtrack.graphs.G1(0)
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
                l_graph = mtrack.graphs.g1_graph.G1(0)
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

                    l_graph.set_orientation(v, ori)
                    
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
            
    def build_line_graph(self, epsilon_lines, use_ori=True, weight_ori=None):
        processed_lines, _ = self.__process_lines(update_ori=use_ori, 
                                                  weight_ori=weight_ori) 

        print "Initialize trees..." 
        if use_ori:
            trees = [KDTree(line_attr["pos_ori"]) for line_attr in processed_lines]
        else:
            trees = [KDTree(line_attr["pos"]) for line_attr in processed_lines]

        line_graph = mtrack.graphs.G1(0)
        line_graph.new_vertex_property("line_id", "int")
        for v_id in range(len(processed_lines)):
            v = line_graph.add_vertex()
            line_graph.set_vertex_property("line_id", v, v_id)

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

                if n_hits>=1 :
                    len_line = float(len(processed_lines[line_id]["v_line"]))
                    len_line_cmp = float(len(processed_lines[line_id_cmp]["v_line"]))
                    line_graph.add_edge(line_id, line_id_cmp)
                    weights.append(n_hits/(len_line + len_line_cmp))

        edge_weight = line_graph.new_edge_property("weight", "float", vals=weights)
        return line_graph, processed_lines

    def get_high_connectivity_cluster(self, epsilon_lines, use_ori, weight_ori, remove_singletons, output_dir):
        line_graph, processed_lines = self.build_line_graph(epsilon_lines, use_ori, weight_ori)
        hcs = line_graph.get_hcs(line_graph, remove_singletons=remove_singletons)
        c = 0
        n = 0
        cluster_list = []
        for cluster in hcs:
            print cluster.get_number_of_vertices()

            line_cluster_dir = os.path.join(output_dir, "line_cluster_c%s/c%s" % (c, n)) 
            if not os.path.exists(line_cluster_dir):
                os.makedirs(line_cluster_dir + "/gt")
 
            l_graph_cluster = []
            for v in cluster.get_vertex_iterator():
                line_id = cluster.get_vertex_property("line_id", v)
                l_graph = processed_lines[line_id]["l_graph"]
                g1_to_nml(l_graph,
                          line_cluster_dir + "/line_%s.nml" % line_id,
                          knossify=True,
                          voxel_size=self.voxel_size)
                l_graph_cluster.append(l_graph)
                #l_graph.save(os.path.join(line_cluster_dir, "gt/line_%s.gt") % line_id)
                n += 1
            
            cluster_list.append(l_graph_cluster)
            combine_knossos_solutions(line_cluster_dir, 
                                      os.path.join(output_dir, 
                                                   "combined/combined_%s.nml" % c), 
                                                   tag="line")
            c += 1

        return cluster_list
            
  
    def fit_nested_line_sbm(self, 
                            epsilon_lines, 
                            output_dir, 
                            use_ori=True, 
                            weight_ori=None):
        
        
        line_graph, processed_lines = self.build_line_graph(epsilon_lines, use_ori, weight_ori)
 
        level_path_list = line_graph.get_sbm(output_dir, nested=True, edge_weights=True)

        if not level_path_list:
            raise Warning("No matching lines")

        print "Combine Lines.."
        n = 0
        l = 0
        for cc_path_list in level_path_list:
            for cc in cc_path_list:
                cc_graph = mtrack.graphs.G1(0)
                cc_graph.load(cc)
            
                line_cluster = mtrack.graphs.G1(0)
            
                line_cluster_dir = os.path.join(output_dir, "line_cluster_l%s/c%s" % (l, n)) 
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

                combine_knossos_solutions(line_cluster_dir, 
                                          os.path.join(output_dir, 
                                                       "combined_l%s/combined_%s.nml" % (l, n)), 
                                                       tag="line")
                n += 1
            l += 1

        return 0


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

def baseline_test(prob_map_stack,
                  gs,
                  ps,
                  bounding_box,
                  remove_singletons,
                  distance_threshold,
                  voxel_size=[5.,5.,50.]):

    candidates = extract_candidates(prob_map_stack,
                                    gs,
                                    ps,
                                    voxel_size,
                                    bounding_box=bounding_box,
                                    bs_output_dir=None)

    g1=candidates_to_g1(candidates, voxel_size)
    g1_connected = connect_graph_locally(g1, distance_threshold)
    hcs = g1_connected.get_hcs(g1_connected, remove_singletons=remove_singletons)

    j = 0
    for cluster in hcs:
        g1_to_nml(cluster, "/media/nilsec/m1/gt_mt_data/experiments/candidate_cut/c_{}".format(j))
        j += 1