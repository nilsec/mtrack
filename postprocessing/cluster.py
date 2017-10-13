from evaluation import OptMatch, get_lines, interpolate_nodes
from scipy.ndimage.morphology import generate_binary_structure, binary_dilation, distance_transform_edt
from scipy.ndimage.measurements import label
import os
import numpy as np
import pdb
import pickle
import h5py
import neuroglancer

def view(raw, seg, voxel_size, offset):
    raw = h5py.File(raw)["data/raw"]
    viewer = neuroglancer.Viewer(voxel_size)
    viewer.add(raw, name="raw")
    viewer.add(seg, name="seg", volume_type="segmentation", offset=offset)
    print viewer

def line_segment_distance(line_p0, line_p1, point):
    line_p0 = np.array(line_p0)
    line_p1 = np.array(line_p1)
    point = np.array(point)

    line_vec = line_p1 - line_p0
    w_0 = point - line_p0
    w_1 = point - line_p1

    #Check if point before, after, or parallel to line segment:
    if np.linalg.norm(line_vec) > 0:
        if np.dot(w_0, line_vec) < 0:
            return np.linalg.norm(point - line_p0)
        elif np.dot(w_1, line_vec) > 0:
            return np.linalg.norm(point - line_p1)
        else:
            d = np.linalg.norm(np.cross(w_0, line_vec))/np.linalg.norm(line_vec)
            return d
    else:
        return None


class Cluster(OptMatch):
    def __init__(self, solution_file, chunk_size=None):
        self.solution_file = solution_file
        self.chunk_size = chunk_size
        self.lines_itp = self.__get_lines()
        if chunk_size is not None:
            self.chunks, self.chunk_positions, self.inv_gt_chunk_positions, _ = OptMatch.get_chunks(self, 
                                                                                                    self.lines_itp, 
                                                                                                    chunk_size)

    def __get_lines(self):
        if self.solution_file[-1] == "/":
            line_base_dir = os.path.dirname(self.solution_file[:-1]) + "/lines"
        else:
            line_base_dir = os.path.dirname(self.solution_file) + "/lines"

        rec_line_dir = line_base_dir + "/reconstruction"
        if not os.path.exists(rec_line_dir):
            line_paths = get_lines(self.solution_file, 
                                   rec_line_dir + "/", 
                                   nml=True)
        else:
            line_paths = [rec_line_dir + "/" + f for f in os.listdir(rec_line_dir) if f.endswith(".gt")]

        lines_itp = interpolate_nodes(line_paths, voxel_size=[5.,5.,50.])

        return lines_itp


    def get_distance_map(self, distance_threshold, voxel_size):
        base_chunk_map = {}
        empty_chunks = []

        for l_base in range(len(self.lines_itp)):
            print "line %s/%s" % (l_base, len(self.lines_itp))
            reached_end_base = False
            c_base = 0
            while not reached_end_base:
                try:
                    chunk = self.chunks[(l_base, c_base)]
                    if chunk:
                        line_p0 = np.array(voxel_size) * np.array(chunk[0])
                        line_p1 = np.array(voxel_size) * np.array(chunk[-1])
                    else:
                        empty_chunks.append((l_base, c_base))
                        c_base += 1
                        continue

                except KeyError:
                    reached_end_base = True
                
                base_chunk_map[(l_base, c_base)] = []
                for l_cmp in range(l_base + 1, len(self.lines_itp)):
                    reached_end_cmp = False
                    c_cmp = 0
                    while not reached_end_cmp:
                        try:
                            chunk_pos_vox = self.chunk_positions[(l_cmp, c_cmp)]
                            chunk_pos_phys = np.array(voxel_size) * chunk_pos_vox

                            d_base_cmp = line_segment_distance(line_p0,
                                                               line_p1,
                                                               chunk_pos_phys)
                            if d_base_cmp is not None:
                                if d_base_cmp < distance_threshold:
                                    base_chunk_map[(l_base, c_base)].append([(l_cmp, c_cmp), d_base_cmp])
                            
                            c_cmp += 1
                        except KeyError:
                            reached_end_cmp = True

                c_base += 1 

        return base_chunk_map

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


    def get_overlap(self, epsilon, min_overlap, offset, voxel_size, canvas_shape, save_volume=None, dt=False):
        canvas_base = np.zeros(canvas_shape)
        dilation = np.array(np.floor(epsilon/np.array(voxel_size)), dtype=int)
        
        # This assumes that the x,y res is the same!
        assert(voxel_size[0] == voxel_size[1])
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
                                            sampling=voxel_size[::-1])

                canvas = (canvas <= epsilon).astype(np.uint8)

            canvas_base[bb_min[2]:bb_max[2], bb_min[1]:bb_max[1], bb_min[0]:bb_max[0]] += canvas
            n += 1

        labeled_volume, n_components = label(canvas_base>=min_overlap)
        
        if save_volume is not None:
            f = h5py.File(save_volume, "w")
            group = f.create_group("data")
            group.create_dataset("overlap", data=labeled_volume.astype(np.dtype(np.uint8)))
 
        return labeled_volume.astype(np.uint8), n_components


if __name__ == "__main__":
    test_solution = "/media/nilsec/d0/gt_mt_data/solve_volumes/test_volume_grid1_ps0505_300_399/solution/volume.gt" 
    validation_solution = "/media/nilsec/d0/gt_mt_data/solve_volumes/grid_2/grid_32/solution/volume.gt"
    cluster = Cluster(validation_solution)
    labeled_volume, n_comps = cluster.get_overlap(epsilon=100,
                                                  min_overlap=3, 
                                                  offset=np.array([0,0,300]),
                                                  voxel_size=[5.,5.,50.],
                                                  canvas_shape=[30,1024,1024],
                                                  save_volume="/media/nilsec/d0/gt_mt_data/data/cluster.h5")

    raw_test = "/media/nilsec/d0/gt_mt_data/data/Test/raw.h5"
    raw_validation = "/media/nilsec/d0/gt_mt_data/data/Validation/raw.h5"
    view(raw_validation, labeled_volume, voxel_size=[5.,5.,50.], offset=[0,0,300*50])


