import postprocessing
import preprocessing
import graphs
from dda3 import DDA3
import numpy as np
import pdb

def get_solution_lines(solution_dir, output_dir, nml=False, scale=None):
    gt_solutions = postprocessing.get_solutions(solution_dir, ".gt")
    line_list_gt = []

    positions = []
   
    for sol in gt_solutions:
        g1 = graphs.G1(0)
        g1.load(sol)
        positions.append(g1.get_position_array())

        if scale is not None:
            for v in g1.get_vertex_iterator():
                pos_scaled = np.array(g1.get_position(v))/scale
                pos_scaled += 0.5 # -0.5 actually but since knossos starts at 1, += 0.5
                g1.set_position(v, np.array(pos_scaled))
        
        if g1.get_number_of_vertices() > 0:
            lines = g1.get_components(min_vertices=1, output_folder=output_dir)
            line_list_gt.extend(lines)

    if nml:
        lines_to_nml(line_list_gt, knossify=True)
    
    return line_list_gt

def get_tracing_lines(tracing, output_dir, nml=False):
    g1_tracing = preprocessing.nml_to_g1(tracing, None)

    positions = g1_tracing.get_position_array()

    lines = g1_tracing.get_components(min_vertices=1, output_folder=output_dir)
    
    if nml:
        lines_to_nml(lines, knossify=True)

    return lines

def lines_to_nml(line_list_gt, 
                 knossos=False,
                 voxel=False,
                 knossify=False):

    for line in line_list_gt:
        g1 = graphs.G1(0)
        g1.load(line)

        positions = g1.get_position_array()
        
        preprocessing.g1_to_nml(g1, line.replace(".gt", ".nml"))


        if voxel:
            preprocessing.g1_to_nml(g1, line.replace(".gt", "_vox.nml"), 
                                    voxel=True, 
                                    voxel_size=[5.0, 5.0, 50.0])
        if knossos:
            preprocessing.g1_to_nml(g1, line.replace(".gt", "_kno.nml"), 
                                    knossos=True, 
                                    voxel_size=[5.0, 5.0, 50.0])
        
        if knossify:
            preprocessing.g1_to_nml(g1, line.replace(".gt", "_kfy.nml"), 
                                    knossify=True)

def get_volume(line_list, dimensions, correction=np.array([0,0,0])):
    canvas = np.zeros([dimensions[2], dimensions[1], dimensions[0]])
    
    print "Interpolate Nodes..."
    label = 1
    for line in line_list:
        g1 = graphs.g1_graph.G1(0)
        g1.load(line)
        
        for edge in g1.get_edge_iterator():
            start = np.array(g1.get_position(edge.source()), dtype=int)
            start -= correction
            end = np.array(g1.get_position(edge.target()), dtype=int)
            end -= correction

            dda = DDA3(start, end)
            skeleton_edge = dda.draw()

            for point in skeleton_edge:
                canvas[point[2], point[1], point[0]] = label

        label += 1

    return canvas
    

if __name__ == "__main__":
    solution_dir = "/media/nilsec/d0/gt_mt_data/experiments/benchmark05_it5_solve_1/solution"
    output_dir = "/media/nilsec/d0/gt_mt_data/experiments/benchmark05_it5_solve_1/lines/"
    get_lines(solution_dir, output_dir)
