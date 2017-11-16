import postprocessing
import preprocessing
import graphs
from dda3 import DDA3
import numpy as np
import os
import pdb

def get_lines(volume, output_dir, voxel_size=[5.0,5.0,50.0], nml=False):
    if volume.endswith(".nml"):
        gt = False
        volume = preprocessing.nml_to_g1(volume, None)
    else:
        assert(volume.endswith(".gt"))
        gt = True
        g1 = graphs.G1(0)
        g1.load(volume)
        volume = g1
 
    if gt:
        for v in volume.get_vertex_iterator():
            pos_scaled = np.array([g1.get_position(v)[j]/voxel_size[j] for j in range(3)])
            volume.set_position(v, pos_scaled)
        
    line_paths = volume.get_components(min_vertices=1, 
                                       output_folder=output_dir)

    if nml:
        lines_to_nml(line_paths, knossify=True)

    return line_paths

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

def get_line_list(directory):
    lines = os.listdir(directory)
    lines = [os.path.join(directory, f) for f in lines if f.endswith(".gt")]
    return lines 
