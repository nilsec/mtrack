import numpy as np
from xml.dom import minidom
import os
import sys
import ast

class NodeAttribute:
    def __init__(self, position, id, orientation, partner, identifier):
        self.position = position
        self.id = id
        self.orientation = orientation
        self.partner = partner
        # Custom id needed for compatibility with prior implementation. 
        # Used for remapping node attributes.
        self.identifier = identifier


def g1_to_nml(g1,
              output_file,
              knossos=False,
              voxel=False,
              voxel_size=None):

    if knossos:
        assert(voxel_size is not None)
        graph_id = 3 # knossos format
        
    elif voxel:
        assert(voxel_size is not None)
        graph_id = 2 # voxel format

    else:
        graph_id = 1 # physical format

    doc = minidom.Document()
    annotations_elem = doc.createElement("things")
    doc.appendChild(annotations_elem)


    annotation_elem = doc.createElement("thing")
    build_attributes(annotation_elem, [["id", graph_id]])
    
    nodes_elem = doc.createElement("nodes")
    edges_elem = doc.createElement("edges")

    g1_index_map = g1.get_vertex_index_map()

    for v in g1.get_vertex_iterator():
        node_elem = doc.createElement("node")
        
        # g1 graph is assumed to always contain
        # physical coordinates as scaling
        # is already done in candidate extraction
        position = np.array(g1.get_position(v))
        orientation = np.array(g1.get_orientation(v))
        partner = g1.get_partner(v)
        node_id = g1.get_vertex_id(v, g1_index_map)

        if knossos or voxel:
            position = np.array([position[j]/voxel_size[j] for j in range(3)])
            orientation = np.array([orientation[j]/voxel_size[j] for j in range(3)])

        if knossos:
            position = np.rint(position).astype(int)
            node_id += 1

            if partner != (-1):
                partner += 1

        identifier = node_id

        build_attributes(node_elem, [["x", position[0]],
                                     ["y", position[1]],
                                     ["z", position[2]],
                                     ["id", node_id],
                                     ["orientation", orientation],
                                     ["partner", partner],
                                     ["identifier", identifier]
                                    ])
        
        nodes_elem.appendChild(node_elem)

    for e in g1.get_edge_iterator():
        assert(e.target() != -1)
        assert(e.source() != -1)

        source_id = g1.get_vertex_id(e.source(), g1_index_map)
        target_id = g1.get_vertex_id(e.target(), g1_index_map)

        if knossos:
            source_id += 1
            target_id += 1

        edge_elem = doc.createElement("edge")
        
        build_attributes(edge_elem, [["source", source_id],
                                     ["target", target_id]
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

    print "G1 graph written to {}".format(output_file)

    return output_file

def build_attributes(xml_elem, attributes):
    for attr in attributes:
        try:
            xml_elem.setAttribute(attr[0], str(attr[1]))
        except UnicodeEncodeError:
            xml_elem.setAttribute(attr[0], str(attr[1].encode('ascii', 'replace')))
    return xml_elem


def from_nml(filename):
    doc = minidom.parse(filename)
    annotation_elems = doc.getElementsByTagName("thing")
    node_dic = {}
    edge_list = []
    for annotation_elem in annotation_elems:
        node_elems = annotation_elem.getElementsByTagName("node")
        for node_elem in node_elems:
            #point, id, orientation, partner, identifier = from_node_elem_to_node(node_elem)
            node_attribute = NodeAttribute(*from_node_elem_to_node(node_elem))
            if node_attribute.id in node_dic:
                print 'WARNING: ID already exists'
                break
            else:
                #point_dic[id] = [point, id, orientation, partner, identifier]
                node_dic[node_attribute.id] = node_attribute

        edge_elems = annotation_elem.getElementsByTagName("edge")
        for edge_elem in edge_elems:
            (source_ID, target_ID) = parse_attributes(edge_elem, [["source", int], ["target", int]])
            edge_list.append([source_ID, target_ID])

    return node_dic, edge_list

def from_node_elem_to_node(node_elem):
    [x, y, z, ID, radius, orientation, partner, identifier] =\
        parse_attributes(node_elem, 
                        [
                            ["x", float], 
                            ["y", float], 
                            ["z", float], 
                            ["id", int], 
                            ["radius", float], 
                            ["orientation", str], 
                            ["partner", int], 
                            ["identifier", int]
                        ]
                        )

    point = np.array([x, y, z])
    
    if orientation is not None:
        orientation = orientation.replace("[", " ").replace("]", " ")
        split = orientation.split(" ")
        rec_ori = []
        for i in split:
            try:
                rec_ori.append(float(i))
            except:
                continue

        orientation = np.array(rec_ori)
        for j in xrange(np.shape(orientation)[0]):
            orientation[j] = float(orientation[j])
    else:
        orientation = np.array([0., 0., 0.])

    return point, ID, np.array(orientation), partner, identifier




def parse_attributes(xml_elem, parse_input):
    parse_output = []
    attributes = xml_elem.attributes
    for x in parse_input:
        try:
            parse_output.append(x[1](attributes[x[0]].value))
        except KeyError:
            parse_output.append(None)
    return parse_output


if __name__ == "__main__":
    from_nml("test_physical_connected.nml")
    

