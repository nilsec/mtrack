from graph import G
from start_edge import StartEdge 
import numpy as np

class G1(G):
    START_EDGE = StartEdge()
    
    def __init__(self, N):
        G.__init__(self, N)

        G.new_vertex_property(self, "orientation", dtype="vector<double>")
        G.new_vertex_property(self, "position", dtype="vector<double>")
        G.new_vertex_property(self, "partner", dtype="long long")

        # NOTE: Start/End dummy node and edge are implicit
        
    def add_edge(self, u, v):
        G.add_edge(self, u, v)
        
    def set_orientation(self, u, value):
        assert(type(value) == np.ndarray)
        assert(len(value) == 3)
        assert(value.ndim == 1)

        G.set_vertex_property(self, "orientation", u, value)
        
    def get_orientation(self, u):
        orientations = G.get_vertex_property(self, "orientation")
        return orientations[u]

    def set_position(self, u, value):
        assert(type(value) == np.ndarray)
        assert(len(value) == 3)
        assert(value.ndim == 1)
 
        G.set_vertex_property(self, "position", u, value)

    def get_position(self, u):
        positions = G.get_vertex_property(self, "position")
        return positions[u]

    def set_partner(self, u, value):
        assert(type(value) == int)
        assert(value < G.get_number_of_vertices(self))
        
        G.set_vertex_property(self, "partner", u, value)
        
    def get_partner(self, u):
        partner = G.get_vertex_property(self, "partner")
        return partner[u]
