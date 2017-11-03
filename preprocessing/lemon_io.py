import numpy as np
import os
import graphs

def g1_to_lemon(g1,
                output_file):
    
    lgf_file = open(output_file, "w+")
    lgf_file.write("@nodes\nlabel position\n")
    label = 0
    # Define a mapping from old 
    # to new labels/node ids:
    old_to_new = {}
    for v in g1.get_vertex_iterator():
        old_to_new[int(v)] = label
        pos = np.array(g1.get_position(v))
        print "WARNIG: CHANGE RESOLUTION IN VOLUME GRAPH & CHECK XYZ ZYX CONVENTION!"
        assert(np.all(pos == np.array(pos, dtype=int))) # Check if we are in voxel space
        lgf_file.write("{}      {},{},{}\n".format(label, int(pos[0]), int(pos[1]), int(pos[2])))
        label += 1

    lgf_file.write("\n")

    lgf_file.write("@edges\n")
    lgf_file.write("      label\n")
    label = 0
    for e in g1.get_edge_iterator():
        source = old_to_new[int(e.source())]
        target = old_to_new[int(e.target())]
        lgf_file.write("{}  {}  {}\n".format(source, target, label))
        label += 1


if __name__ == "__main__":
    g = graphs.G1(4)
    for j in range(4):
        g.set_position(j, np.array([j, 0., 0.]))
        g.add_edge(j, (j+1)%4)

    g1_to_lemon(g, "lemon.lgf")
        
        
        
    
        
        
        
