import graph_tool.all as gt
import numpy as np

class G1:
    """
    Wrapper class for the G1 graph.
    If we decide to change the backend in the future
    we only need to change the corresponding methods.
    """
    def __init__(self, N):
        self.N = N
        self.g = gt.Graph(directed=False)
        self.g.add_vertex(N)

    def get_number_of_edges(self):
        return self.g.num_edges()

    def get_number_of_vertices(self):
        return self.g.num_vertices()

    def get_vertex(self, u):
        return self.g.vertex(u)
    
    def add_vertex(self):
        """
        Add vertex and return 
        vertex descriptor.
        """
        u = self.g.add_vertex()
        return u

    def get_edge(self, edge_key):
        #TODO: Check if this is the most efficient way.
        """
        FOR CONSISTENCY: ALL EDGE OPERATIONS
        RELY ONLY ON THE EDGE MATRIX.
        Returns edge in form np.array(u, v, id) by
        providing a tuple of the form (u, v) or an edge id.
        """

        if isinstance(edge_key, tuple):
            return self.__get_edge_by_vertex(edge_key)
        elif isinstance(edge_key, int):
            return self.__get_edge_by_id(edge_key)

    def __get_edge_by_vertex(self, (u, v)):
        for e in self.get_edge_matrix():
            if np.all(np.array([e[0], e[1]]) == np.array([u, v])):
                return e
        
        return None

    def __get_edge_by_id(self, edge_id):
        for e in self.get_edge_matrix():
            if e[2] == edge_id:
                return e
 
        return None

    def add_edge(self, (u, v), reindex=False):
        """
        Add edge and return 
        edge descriptor of the form
        np.array(u, v, id)
        """
        e = self.g.add_edge(u, v)
        
        if reindex:
            self.g.reindex_edges()
        
        return self.__get_edge_by_vertex((e.source(), e.target()))
        
    def get_edge_matrix(self):
        """
        graph_tool.get_edges()
        Return a numpy.ndarray containing the edges. The shape of
        the array will be (E, 3) where E is the number of edges and
        each line will contain the source, target and index of an edge.
        """
        return self.g.get_edges()

    def get_neighbour_nodes(self, u):
        """
        Returns sorted vertex id's.
        """
        return np.sort(self.g.get_out_neighbours(u))

    def get_incident_edges(self, u):
        """
        Returns incident edges sorted by id.
        """
        edges = self.g.get_out_edges(u)
        return np.array(sorted(edges, key=lambda x: x[2]))

    def random_init(self, deg_sampler, args):
        """
        Initialize random graph with custom 
        degree sampler.
        """
        self.g = gt.random_graph(N=self.N,
                                 deg_sampler=lambda: deg_sampler(**args),
                                 directed=False,
                                 parallel_edges=False,
                                 random=True)
