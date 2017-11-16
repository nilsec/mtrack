import os
import sys
sys.path.append(os.path.join('..', '..'))
from xml.dom import minidom
from preprocessing import nml_io
import graphs
import numpy as np


def get_solutions(solution_dir, file_tag):
    files = []

    for dirpath, dirnames, filenames in os.walk(solution_dir):
        for f in filenames:
            if file_tag in f:
                files.append(os.path.join(dirpath, f))

    return sorted(files)

def combine_knossos_solutions(solution_dir, output_file, tag=None):

    if tag is None:
        tag = "kno"

    files = get_solutions(solution_dir, tag)
    
    doc = minidom.Document()
    annotations_elem = doc.createElement("things")
    doc.appendChild(annotations_elem)

    annotation_elem = doc.createElement("thing")
    nml_io.build_attributes(annotation_elem, [["id", 3]])
    
    nodes_elem = doc.createElement("nodes")
    edges_elem = doc.createElement("edges")

    for f in files:
        node_dic, edge_list = nml_io.from_nml(f)
        
        for node_id, node_attribute in node_dic.iteritems():
            node_elem = doc.createElement("node")
            position = node_attribute.position
            orientation = node_attribute.orientation
            partner = node_attribute.partner
            identifier = node_attribute.identifier

            nml_io.build_attributes(node_elem, [["x", int(position[0])],
                                     ["y", int(position[1])],
                                     ["z", int(position[2])],
                                     ["id", node_id],
                                     ["orientation", orientation],
                                     ["partner", partner],
                                     ["identifier", identifier]
                                         ])
        
            nodes_elem.appendChild(node_elem)

        for edge in edge_list:
            edge_elem = doc.createElement("edge")
        
            nml_io.build_attributes(edge_elem, [["source", edge[0]],
                                         ["target", edge[1]]
                                        ])

            edges_elem.appendChild(edge_elem)

    annotation_elem.appendChild(nodes_elem)
    annotation_elem.appendChild(edges_elem)
    annotations_elem.appendChild(annotation_elem)
    doc = doc.toprettyxml()

    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w+") as f:
        f.write(doc)


def combine_gt_solutions(solution_dir, output_file):
    files = get_solutions(solution_dir, ".gt")
    positions = []
    orientations = []
    edges = []
    n_edges = []
    N = []
    index_map = {-1: -1}

    g1_tmp = graphs.g1_graph.G1(0)

    for f in files:
        g1_tmp.load(f)
        #if g1_tmp.get_number_of_vertices() == 0:
        #    continue
        edges.append(g1_tmp.get_edge_array())
        #edges[-1] += sum(N)

 
        index_map.update({v: j + sum(N) for j, v in enumerate(g1_tmp.get_vertex_iterator())})

        N.append(g1_tmp.get_number_of_vertices())
        positions.append(g1_tmp.get_position_array().T)
        orientations.append(g1_tmp.get_orientation_array().T)
               
    N_comb = sum(N)
    positions = np.vstack(positions)
    orientations = np.vstack(orientations)
    edges = np.delete(np.vstack(edges), 2, 1)

    
    g1_comb = graphs.g1_graph.G1(N_comb)
    for v in range(N_comb):
        g1_comb.set_position(v, positions[v])
        g1_comb.set_orientation(v, orientations[v])

    index_map = np.vectorize(index_map.get)
    edges = index_map(edges)

    g1_comb.add_edge_list(edges)
    g1_comb.save(output_file)
    return g1_comb
