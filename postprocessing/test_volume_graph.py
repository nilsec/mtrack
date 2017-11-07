import graphs
import os
import h5py
import skeletopyze
import pdb
import numpy as np
import itertools
import preprocessing

def volume_to_g1(volume):
    g1 = graphs.G1(0)

    nb = [[-1,0,0],
          [0,-1,0],
          [0,0,-1],
          [-1,-1,0],
          [-1, 0, -1],
          [0, -1, -1],
          [1,0,0],
          [0,1,0],
          [0,0,1],
          [1,1,0],
          [1,0,1],
          [0,1,1],
          [-1, 1, 0],
          [-1, 0, 1],
          [0, -1, 1],
          [1, -1, 0],
          [0, 1, -1],
          [1, 0, -1],
          [1,1,1],
          [1, 1,-1],
          [1,-1,1],
          [-1,1,1],
          [-1,-1,-1],
          [-1,-1, 1],
          [-1, 1, -1],
          [1, -1, -1]]

    nb = set(tuple(j) for j in nb)
    
    dim = np.shape(volume)
    for x,y,z in itertools.product(range(dim[2]), range(dim[1]), range(dim[0])):
        if volume[z,y,x] == 1:
            v = g1.add_vertex()
            g1.set_position(v, np.array([x,y,z]))

    for v1 in g1.get_vertex_iterator():
        for u1 in g1.get_vertex_iterator():
            v_pos = np.array(g1.get_position(v1))
            u_pos = np.array(g1.get_position(u1))
            if int(u1) > int(v1):
                if tuple(v_pos - u_pos) in nb:
                    g1.add_edge(u1, v1)

    preprocessing.g1_to_lemon(g1, "volume.lgf")
    
    for v in g1.get_vertex_iterator():
        print len(g1.get_neighbour_nodes(v))

def get_tube(dim, radius):
    canvas = np.zeros(dim)

    for z in range(dim[0]):
        for x,y in itertools.product(range(int(np.floor(dim[2]/2)) - radius, int(np.ceil(dim[2]/2)) + radius),
                                     range(int(np.floor(dim[1]/2)) - radius, int(np.ceil(dim[1]/2)) + radius)):
            canvas[z,y,x] = 1

    return canvas

def skeletonize(volume_graph):
    params = skeletopyze.Parameters()
    b = skeletopyze.get_sk_graph(volume_graph, params)
    print("Skeleton contains nodes:")
    for n in b.nodes():
        print str(n) + ": " + "(%d, %d, %d), diameter %f"%(b.locations(n).x(), b.locations(n).y(), b.locations(n).z(), b.diameters(n))

    print("Skeleton contains edges:")
    for e in b.edges():
        print (e.u, e.v)

def skeletonize_volume(volume_path, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    res = skeletopyze.point_f3()
    res.__setitem__(0, 5.)
    res.__setitem__(1, 5.)
    res.__setitem__(2, 50.)
 
    params = skeletopyze.Parameters()
    f = h5py.File(volume_path)
    volume = f["data/overlap"].value
    b = skeletopyze.get_skeleton_graph(volume, params, res)

    skeleton = graphs.G1(0)
    for n in b.nodes():
        v = skeleton.add_vertex()
        skeleton.set_position(v, np.array([b.locations(n).x(), b.locations(n).y(), b.locations(n).z() + 300]))
    for e in b.edges():
        skeleton.add_edge(e.u, e.v)

    preprocessing.g1_to_nml(skeleton, output_path, knossify=True)

def skeletonize_volumes(base_dir, output_dir):
    volumes = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]

    n = 0
    for volume in volumes:
        skeletonize_volume(volume, os.path.join(output_dir, "skeleton_{}.nml".format(n)))
        n += 1

if __name__ == "__main__":
    #tube = get_tube([10,10,10], 3)
    #volume_to_g1(tube)
    #skeletonize("volume.lgf")
    #skeletonize_volume("/media/nilsec/m1/gt_mt_data/experiments/lv_clustering/hsc/test_grid1_el100_wo900_rs0/volumes/lvcluster_0.h5",)
    skeletonize_volumes("/media/nilsec/m1/gt_mt_data/experiments/lv_clustering/hsc/test_grid32_el75_wo1500_rs1/volumes", "./skeletons32_75_1500_1")
