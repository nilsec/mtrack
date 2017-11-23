import numpy as np
import os
from xml.dom import minidom

from mtrack.preprocessing import nml_io
import mtrack.graphs


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

def combine_gt_graphs(graph_list, prop_vp=None, prop_vp_dtype=None):
    positions = []
    orientations = []
    edges = []
    n_edges = []

    if prop_vp is not None:
        add_vp = []

    N = []
    index_map = {-1: -1}

    for g in graph_list:
        if isinstance(g, str):
            g_tmp = mtrack.graphs.G1(0)
            g_tmp.load(g)
            g = g_tmp

        index_map.update({v: j + sum(N) for j, v in enumerate(g.get_vertex_iterator())})
        index_map_get = np.vectorize(index_map.get)
    
        edges.append(index_map_get(g.get_edge_array())) 
        N.append(g.get_number_of_vertices())
        positions.append(g.get_position_array().T)
        orientations.append(g.get_orientation_array().T)
        if prop_vp is not None:
            add_vp.append(g.get_vertex_property(prop_vp).a)
               
    N_comb = sum(N)
    positions = np.vstack(positions)
    orientations = np.vstack(orientations)
    if prop_vp is not None:
        add_vp = np.hstack(add_vp)

    edges = np.delete(np.vstack(edges), 2, 1)
    
    g1_comb = mtrack.graphs.g1_graph.G1(N_comb)
    if prop_vp is not None:
        vp = g1_comb.new_vertex_property(prop_vp, prop_vp_dtype)

    for v in range(N_comb):
        g1_comb.set_position(v, positions[v])
        g1_comb.set_orientation(v, orientations[v])
        if prop_vp is not None:
            g1_comb.set_vertex_property(prop_vp, v, add_vp[v])            
    
    g1_comb.add_edge_list(edges)
    return g1_comb    


def combine_gt_solutions(solution_dir, output_file):
    files = get_solutions(solution_dir, ".gt")
    
    combined_graph = combine_gt_graphs(files)

    combined_graph.save(output_file)
    return combined_graph
