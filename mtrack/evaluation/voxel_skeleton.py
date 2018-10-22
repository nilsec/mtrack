import numpy as np

from mtrack.evaluation.dda3 import DDA3
from mtrack.graphs import G1
from mtrack.preprocessing import g1_to_nml

class VoxelSkeleton(object):
    def __init__(self, g1_cc, voxel_size, subsample=1, verbose=False):
        """
        Interpolate a g1 graph connected component
        in voxel space and generate a graph on a voxel 
        grid.

        Scaling only influencec the direction of interpolation
        but not the actual saved positions. These remain in voxel space.
        """
        self.g1_cc = g1_cc
        self.scaling = voxel_size
        self.verbose = verbose
        self.subsample = subsample
        self.points_unique, self.edge_to_line = self.__generate(scaling=voxel_size)
        self.voxel_skeleton = self.__to_graph(self.points_unique, self.edge_to_line)
        if subsample > 1:
            self.voxel_skeleton = self.__subsample(subsample, self.voxel_skeleton)

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

        return points, edge_to_line


    def __to_graph(self, points_unique, edge_to_line):
        """
        Convert point interpolation to
        gt graph to preserve neighborhood
        information.
        """
        if self.verbose:
            print("Initialize interpolation graph...")
        
        # Remove doubly counted vertices found at start/end of each edge
        double_vertices = len(edge_to_line) - 1 
        g1_interpolated = G1(len(points_unique) - double_vertices)

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

    def __subsample(self, subsample, path):
        """
        Requires the graph to be a path.
        """

        # Identify start/end vertex:
        subsample_mask = path.new_vertex_property("subsample", 
                                                  "bool",
                                                  value=False)
        start_end = []
        for v in path.get_vertex_iterator():
            if len(path.get_incident_edges(v)) == 1:
                start_end.append(v)

        # Walk across neighbours:
        v0 = start_end[0]
        n = 0

        visited = set([int(v0)])
        selected = []
        while not int(v0) == int(start_end[1]):
            # Add every subsample
            if n % subsample == 0:
                subsample_mask[int(v0)] = True
                selected.append(int(v0))
            else:
                subsample_mask[int(v0)] = False

            visited.add(int(v0))

            in_edges = path.get_incident_edges(v0)

            try: 
                # Can fail at start/end
                nb_vertices = set([int(in_edges[0].source()),
                                   int(in_edges[0].target()),
                                   int(in_edges[1].source()),
                                   int(in_edges[1].target())])
            except IndexError:
                nb_vertices = set([int(in_edges[0].source()),
                                   int(in_edges[0].target())])

            nb_vertices = nb_vertices.difference(visited)
            assert(len(nb_vertices) == 1)
            v0 = list(nb_vertices)[0]
            n += 1
            
        subsample_mask[start_end[1]] = True
        selected.append(start_end[1])

        path.set_vertex_filter(subsample_mask)
        for i in range(len(selected) - 1):
            path.add_edge(selected[i], selected[i+1])
        
        return path
