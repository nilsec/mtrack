import numpy as np
from g1_graph import G1
from g2_graph import G2

class GraphConverter:
    def __init__(self, g1):
        self.g1 = g1

    def check_edge_partner_conflict(self, e1, e2):
        """
        Check if the formation of a G2 node that
        contains two G1 edges violates
        conflicts on the G1 graph.
        This type of error would be uncorrectable
        later on.
        """
        
        if e2 != G1.START_EDGE:
            vertices_in_e2 = [e2.source(), e2.target()]
        else:
            # There can be no conflict in edges
            # if one is a start edge
            return False

        if e1 != G1.START_EDGE:
            vertices_in_e1 = [e1.source(), e1.target()]
            
            for v in vertices_in_e1:
                partner_vertex_v = self.g1.get_partner(v)

                if partner_vertex_v != (-1): # -1 encodes no partner
                    if partner_vertex_v in vertices_in_e2:
                        return True
        return False

    def get_mapping(self):
        """
        Generic mapping function. Works for all types
        of G1 graphs independent of property maps.
        Returns g2 center conflicts and several
        mappings from g1 to g2 attributes.
        """

        g1_edge_index_map = self.g1.get_edge_index_map()
        g1_vertex_index_map = self.g1.get_vertex_index_map()

        g1edge_g2vertices = {self.g1.get_edge_id(e, g1_edge_index_map): 
                             [] for e in self.g1.get_edge_iterator()}
        g1edge_g2vertices[G1.START_EDGE.id()] = []

        g2vertex_g1edges = {}

        # o----o----o    == g2 vertex (the whole thing)
        #      ^         == center g1 vertex of this g2 vertex
        g1_vertex_center = {}
        g1_vertex_center_inverse = {g1_vertex_index_map[v]:\
                                    [] for v in self.g1.get_vertex_iterator()}

        g2_vertices_N = 0
        g2_center_conflicts = []

        # Create one g2 vertex with v as the center node for all v in g1
        for v in self.g1.get_vertex_iterator():
            v_conflicts = []
            g1_incident_edges_v = self.g1.get_incident_edges(v) 
            
            for e1 in g1_incident_edges_v:
                for e2 in g1_incident_edges_v + [G1.START_EDGE]:
                    partner_conflict = self.check_edge_partner_conflict(e1, e2)
               
                    e1_id = self.g1.get_edge_id(e1, g1_edge_index_map) 
                    e2_id = self.g1.get_edge_id(e2, g1_edge_index_map)

                    if (e1_id >= e2_id and e2_id != G1.START_EDGE.id()) or partner_conflict:
                        continue # Do not create g2 vertex that conflicts with itself.

                    # Create g2 vertex
                    g2_vertex_id = g2_vertices_N
                    g2_vertices_N += 1
                    
                    # Fill index maps 
                    g2vertex_g1edges[g2_vertex_id] = (e1, e2)
                    g1edge_g2vertices[e1_id].append(g2_vertex_id)
                    g1edge_g2vertices[e2_id].append(g2_vertex_id)
                    g1_vertex_center[g2_vertex_id] = v
                    g1_vertex_center_inverse[v].append(g2_vertex_id)

                    # All g2 vertices centered around v are in conflict
                    # o----o----o
                    #     v^
                    v_conflicts.append(g2_vertex_id)
            
            if len(v_conflicts) > 1: 
                g2_center_conflicts.append(v_conflicts)

        index_maps = {"g2vertex_g1edges": g2vertex_g1edges,
                      "g1edge_g2vertices": g1edge_g2vertices,
                      "g1_vertex_center": g1_vertex_center,
                      "g1_vertex_center_inverse": g1_vertex_center_inverse}
    
        return g2_vertices_N, g2_center_conflicts, index_maps

    def get_partner_conflicts(self, g2, g1_vertex_center_inverse):
        """
        Get conflicts on the G2 graph that are derived
        from the (possibly empty) partner vertex
        property of the G1 graph.
        """
        g2_partner_conflicts = []
        
        for v in self.g1.get_vertex_iterator():
            # Get all g2 nodes that have v
            # as a center node
            g2_vertex = g1_vertex_center_inverse[v]
            g1_partner = self.g1.get_partner(v)

            # Check which g2 nodes contain partner
            # nodes of v, those are mutually exclusive:
            if g1_partner > v and g1_partner != (-1): # -1 encodes no partner
                g2_partner = g1_vertex_center_inverse[g1_partner]
                
                g2_partner_conflicts.append(g2_partner + g2_vertex)

        return g2_partner_conflicts

    def get_continuation_constraints(self, g1edge_g2vertices, g1_vertex_center):
        continuation_constraints = []       
        edge_index_map = self.g1.get_edge_index_map()       
 
        for e in self.g1.get_edge_iterator():
            # Get all g2 nodes that contain the g1 edge e
            g2_vertices_e = g1edge_g2vertices[self.g1.get_edge_id(e, edge_index_map)]
            
            # Divide those in two groups representing
            # those g2 nodes that are to the "left"
            # and to the "right" of the g1 edge e.
            g1_vertex_l = e.source()
            g1_vertex_r = e.target()

            g2_vertices_l = [v for v in g2_vertices_e if g1_vertex_center[v] == g1_vertex_l]
            g2_vertices_r = [v for v in g2_vertices_e if g1_vertex_center[v] == g1_vertex_r]

            assert(len(g2_vertices_l) + len(g2_vertices_r) == len(g2_vertices_e))

            g2_rl = {"g2_vertices_l": g2_vertices_l, "g2_vertices_r": g2_vertices_r}
            continuation_constraints.append(g2_rl)

        return continuation_constraints

    def get_g2_graph(self):
        # Get Number of Nodes, Index maps and derived conflicts:
        g2_vertices_N, g2_center_conflicts, index_maps = self.get_mapping()
        
        # Create G2 Graph:
        g2 = G2(g2_vertices_N)
        
        # Propagate partner conflicts up (from g1 to g2):
        g2_partner_conflicts = self.get_partner_conflicts(g2, 
                                                          index_maps["g1_vertex_center_inverse"])
        
        # Get continuation constraints to ensure no isolated 
        # line segments without start and end node:
        g2_continuation_constraints =\
                self.get_continuation_constraints(index_maps["g1edge_g2vertices"],
                                                  index_maps["g1_vertex_center"])

        # Initialize the g2 graph:
        for conflict in g2_center_conflicts + g2_partner_conflicts:
            g2.add_conflict(conflict)

        for constraint in g2_continuation_constraints:
            g2.add_sum_constraint(constraint["g2_vertices_l"],
                                  constraint["g2_vertices_r"])

        return g2, index_maps