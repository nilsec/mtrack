import os
import sys
sys.path.append(os.path.join('..', '..'))
from xml.dom import minidom
from preprocessing import nml_io


def get_solutions(solution_dir, file_tag):
    files = []

    for dirpath, dirnames, filenames in os.walk(solution_dir):
        for f in filenames:
            if file_tag in f:
                files.append(os.path.join(dirpath, f))

    return sorted(files)

def combine_knossos_solutions(solution_dir, output_file):
    files = get_solutions(solution_dir, "kno")
    
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
 

if __name__ == "__main__":
    sol_89 = "/media/nilsec/d0/gt_mt_data/experiments/validation_solve_dt89" 
    combine_knossos_solutions(sol_89, "./val_89.nml")
