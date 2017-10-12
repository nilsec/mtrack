from evaluation import OptMatch, get_lines, interpolate_nodes
import os
import numpy as np
import pdb


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
    def __init__(self, chunk_size, solution_file):
        self.solution_file = solution_file
        self.chunk_size = chunk_size
        self.lines_itp = self.__get_lines()
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


    def run(self, distance_threshold, voxel_size):
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

    def run_2(self, epsilon, voxel_size):
        


if __name__ == "__main__":
    cluster = Cluster(10, "/media/nilsec/d0/gt_mt_data/solve_volumes/test_volume_grid1_ps0505_300_399/solution/volume.gt")
    base_chunk_map = cluster.run(distance_threshold=100, 
                                 voxel_size=[5.,5.,50.])

    print base_chunk_map
