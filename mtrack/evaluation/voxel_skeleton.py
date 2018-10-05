import numpy as np

from mtrack.evaluation.dda3 import DDA3
from mtrack.graphs import G1


class VoxelSkeleton(object):
    def __init__(self, g1_cc, voxel_size, verbose=False):
        """
        Interpolate a g1 graph connected component
        in voxel space and generate a graph on a voxel 
        grid.
        """
        self.g1_cc = g1_cc
        self.scaling = voxel_size
        self.verbose = verbose
        self.points_unique, self.edge_to_line = self.__generate(scaling=voxel_size)
        self.voxel_skeleton = self.__to_graph(self.points_unique, self.edge_to_line)

    def get_graph(self):
        return self.voxel_skeleton

    def get_points(self):
        return self.points_unique

    def __generate(self, scaling):
        """
        Interpolate each edge linearly 
        in voxel space with appropriate
        scaling.
        """
        if self.verbose:
            print("Interpolate edges...")

        points = []
        edge_to_line = {}
        for e in self.g1_cc.get_edge_iterator():
            start = np.array(self.g1_cc.get_position(e.source()))
            assert(np.all(start.astype(int) == start))
            start = start.astype(int)

            end = np.array(self.g1_cc.get_position(e.target()))
            assert(np.all(end.astype(int) == end))
            end = end.astype(int)

            dda = DDA3(start, end, self.scaling)
            line = dda.draw()
                
            points.extend(line)
            edge_to_line[e] = line

        points_unique = np.unique(points, axis=0) 
        return points_unique, edge_to_line


    def __to_graph(self, points_unique, edge_to_line):
        """
        Convert point interpolation to
        gt graph to preserve neighborhood
        information.
        """
        if self.verbose:
            print("Initialize interpolation graph...")
        
        g1_interpolated = G1(len(points_unique))

        """
        Initialize original cc vertices
        as fixed points.
        """
        vint = 0
        pos_to_vint = {}
        for v in self.g1_cc.get_vertex_iterator():
            v_pos = np.array(self.g1_cc.get_position(v))
            g1_interpolated.set_position(vint, v_pos)
            pos_to_vint[tuple(v_pos)] = vint
            vint += 1

        """
        Add edge interpolation in between each 
        fixed tree vertex.
        """
        for e in self.g1_cc.get_edge_iterator():
            points = edge_to_line[e]
            start_vertex = pos_to_vint[tuple(points[0])]
            end_vertex = pos_to_vint[tuple(points[-1])]
            for pos in points[1:-1]:
                g1_interpolated.add_edge(start_vertex, vint)
                g1_interpolated.set_position(vint, pos)
                start_vertex = vint
                vint += 1

            g1_interpolated.add_edge(start_vertex, end_vertex)

        return g1_interpolated
