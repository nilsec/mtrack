from graph import G
from start_edge import StartEdge 
import numpy as np
import mt_utils
import os
import matplotlib.pyplot as plt
from numpy.core.umath_tests import inner1d
import itertools
import pdb

class G1(G):
    START_EDGE = StartEdge()
    
    def __init__(self, N, G_in=None):
        G.__init__(self, N, G_in=G_in)

        if G_in is None:
            G.new_vertex_property(self, "orientation", dtype="vector<double>")
            G.new_vertex_property(self, "position", dtype="vector<double>")
            G.new_vertex_property(self, "partner", dtype="long long")
        
            for v in G.get_vertex_iterator(self):
                self.set_partner(v, -1) 
        

        # NOTE: Start/End dummy node and edge are implicit
        self.edge_index_map = G.get_edge_index_map(self)
 
        
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

    def get_orientation_array(self, u):
        return G.get_vertex_property(self, "orientation").get_2d_array([0,1,2])

    def set_position(self, u, value):
        assert(type(value) == np.ndarray)
        assert(len(value) == 3)
        assert(value.ndim == 1)
 
        G.set_vertex_property(self, "position", u, value)

    def get_position(self, u):
        positions = G.get_vertex_property(self, "position")
        return positions[u]

    def get_position_array(self):
        return G.get_vertex_property(self, "position").get_2d_array([0,1,2])

    def set_partner(self, u, value):
        assert(type(value) == int)
        assert(value < G.get_number_of_vertices(self))
        
        G.set_vertex_property(self, "partner", u, value)
        
    def get_partner(self, u):
        partner = G.get_vertex_property(self, "partner")
        return partner[u]

    def get_edge_id(self, e, edge_index_map):
        """
        Workaround because of start edge
        """
        
        if isinstance(e, StartEdge):
            return e.id()
        else:
            return edge_index_map[e]

    def get_vertex_id(self, v, vertex_index_map):
        if v == -1:
            return -1
        else:
            return vertex_index_map[v]

    def get_vertex_cost(self):
        vertex_cost = {}
        for v in G.get_vertex_iterator(self):
            vertex_cost[v] = 0.0

        #Start Vertex
        vertex_cost[-1] = 0.0

        return vertex_cost
            
    def get_edge_cost(self, distance_factor, orientation_factor, start_edge_prior):
        edge_array = G.get_edge_array(self)
        edge_cost_tot = {}
        
        edge_cost_tot[self.START_EDGE] = 2.0 * start_edge_prior
        
        for e in edge_array:
            positions = [np.array(self.get_position(e[0])), 
                         np.array(self.get_position(e[1]))]
            orientations = [np.array(self.get_orientation(e[0])), 
                            np.array(self.get_orientation(e[1]))]

            d = np.linalg.norm(positions[0] - positions[1])
            d_cost = distance_factor * d


            orientation_angles = mt_utils.get_orientation_angle(positions,
                                                                orientations)

            o_cost = orientation_factor * sum(orientation_angles)
            
            edge_cost_tot[G.get_edge(self, e[0], e[1])] = d_cost + o_cost

        return edge_cost_tot

    def get_edge_cost_opt(self, distance_factor, orientation_factor, start_edge_prior):
        edge_array = G.get_edge_array(self) 
        pos_array = self.get_position_array()
        orientation_array = self.get_orientation_array()
 
        print edge_array
        return 0

    def get_edge_combination_cost_old(self, comb_angle_factor, comb_angle_prior = 0.0):
        edge_combination_cost = {}
        edge_index_map = G.get_edge_index_map(self)

        for v in G.get_vertex_iterator(self):
            incident_edges = G.get_incident_edges(self, v)
            
            for e1 in incident_edges:
                for e2 in incident_edges + [self.START_EDGE]:
                    e1_id = self.get_edge_id(e1, edge_index_map)
                    e2_id = self.get_edge_id(e2, edge_index_map)

                    if e1_id >= e2_id and e2_id != self.START_EDGE.id():
                        continue

                    if e2_id == self.START_EDGE.id():
                        edge_combination_cost[(e1, e2)] = comb_angle_prior

                    else:
                        middle_vertex = v
                        e1_tuple = [e1.source(), e1.target()]
                        e2_tuple = [e2.source(), e2.target()]

                        assert(middle_vertex in e1_tuple)
                        assert(middle_vertex in e2_tuple)

                        e1_middle_index = e1_tuple.index(middle_vertex)
                        e2_middle_index = e2_tuple.index(middle_vertex)

                        v_middle_pos = self.get_position(e1_tuple[e1_middle_index]) 
                        v1_pos = np.array(self.get_position(e1_tuple[int(not e1_middle_index)]))
                        v2_pos = np.array(self.get_position(e2_tuple[int(not e2_middle_index)]))

                        vector_1 = v1_pos - v_middle_pos
                        vector_2 = v2_pos - v_middle_pos
    
                        comb_angle = mt_utils.get_spanning_angle(vector_1, vector_2)
                        assert((comb_angle <= np.pi) and (comb_angle >= 0)) 
                        comb_angle = np.pi - comb_angle

                        edge_combination_cost[(e1, e2)] = (comb_angle * comb_angle_factor)**2

        return edge_combination_cost


    def get_edge_combination_cost_1(self, comb_angle_factor):
        edge_combination_cost = {}
        edge_index_map = G.get_edge_index_map(self)
        middle_indices = []
        end_indices = []
        edges = []

        for v in G.get_vertex_iterator(self):
            incident_edges = G.get_incident_edges(self, v)
            
            for e1 in incident_edges:
                for e2 in incident_edges + [self.START_EDGE]:
                    e1_id = self.get_edge_id(e1, edge_index_map)
                    e2_id = self.get_edge_id(e2, edge_index_map)

                    if e1_id >= e2_id and e2_id != self.START_EDGE.id():
                        continue

                    if e2_id == self.START_EDGE.id():
                        continue

                    else:
                        middle_vertex = int(v)
                        middle_indices.extend([middle_vertex, middle_vertex])

                        end_vertices = set([int(e1.source()), 
                                            int(e1.target()), 
                                            int(e2.source()), 
                                            int(e2.target())])

                        end_vertices.remove(middle_vertex)
                        end_indices.extend(list(end_vertices))
                        edges.append((e1, e2))

        pos_array = self.get_position_array()
        end_indices = np.array(end_indices)
        middle_indices = np.array(middle_indices)

        v = (pos_array[:, end_indices] - pos_array[:, middle_indices]).T
        norm = np.sum(np.abs(v)**2, axis=-1)**(1./2.)
        u = v/norm[:,None]
        angles = np.arccos(np.clip(inner1d(u[::2], u[1::2]), -1.0, 1.0))
        angles = np.pi - angles
        cost = (angles * comb_angle_factor)**2

        return dict(itertools.izip(edges, cost))

    def get_edge_combination_cost(self, comb_angle_factor, comb_angle_prior=0.0):
        edge_index_map = G.get_edge_index_map(self)
        """
        Only collect the indices for each edge combination in
        the loop and perform cost calculation later in vectorized 
        form.
        """
        middle_indices = []
        end_indices = []
        edges = []
        cost = []
        prior_cost = {}
        """
        Problem: The indices are derived from the vertices in the graph.
        The graphs are filtered versions of a bigger graph where the
        vertices have not been enumerated newly. Thus we expect 
        vertices to have random indices, not corresponding to 
        the number of vertices in the sub graph. If we try to acces
        the position matrix with these indices we get out of bounds
        errors because the position matrix has only the entries 
        of the filtered subgraph. We need to map vertex indices
        in the range [0, N_vertices_subgraph - 1]
        """

        index_map = {}

        for n, v in enumerate(G.get_vertex_iterator(self)):
            incident_edges = G.get_incident_edges(self, v)
            index_map[v] = n 
            
            for e1 in incident_edges:
                for e2 in incident_edges + [self.START_EDGE]:
                    e1_id = self.get_edge_id(e1, edge_index_map)
                    e2_id = self.get_edge_id(e2, edge_index_map)

                    if e1_id >= e2_id and e2_id != self.START_EDGE.id():
                        continue

                    if e2_id == self.START_EDGE.id():
                        """
                        Always append edges together with cost
                        s.t. zip(edges, cost) is a correct mapping
                        of edges to cost.
                        """
                        prior_cost[(e1, e2)] = comb_angle_prior
 
                    else:
                        """
                        Here we only save indices. How to secure
                        a proper matching between cost that we calculate
                        later and the corresponding edges?

                        1. Middle vertex is v -> index_map[v] in [0, N-1]
                        2. Append the middle_vertex twice to a list.
                        3. Append the distinct end vertices (2) to end vertices
                        4. Append the corresponding edges to a list.

                        -> We end up with 3 lists of the following form:
                        edges = [(e1, e2), (e3, e4), ...]
                        m_ind = [ m1, m1 ,  m2, m2 , ...]
                        e_ind = [ v1, v2 ,  v3, v4 , ...]
                        p_arr = [ p(m1)  ,   p(m2) , ...]
                        index_map: m1 -> 0, m2 -> 1, m3 -> 2, ...
                            --> p_arr[m1] = p(m1), p_arr[m2] = p(m2) 
                        """
                        middle_vertex = int(v)
                        middle_indices.extend([middle_vertex, middle_vertex])

                        end_vertices = set([int(e1.source()), 
                                            int(e1.target()), 
                                            int(e2.source()), 
                                            int(e2.target())])

                        end_vertices.remove(middle_vertex)
                        end_indices.extend(list(end_vertices))
                        edges.append((e1, e2))
                        """
                        (e1, e2) -> end_indices, middle_indices
                        """

        if middle_indices:
            pos_array = self.get_position_array()
            end_indices = np.array([index_map[v] for v in end_indices])
            middle_indices = np.array([index_map[v] for v in middle_indices])

            v = (pos_array[:, end_indices] - pos_array[:, middle_indices]).T
            norm = np.sum(np.abs(v)**2, axis=-1)**(1./2.)
            u = v/norm[:,None]
            angles = np.arccos(np.clip(inner1d(u[::2], u[1::2]), -1.0, 1.0))
            angles = np.pi - angles
            cost = cost + list((angles * comb_angle_factor)**2)

        edge_combination_cost = dict(itertools.izip(edges, cost))
        edge_combination_cost.update(prior_cost)
 
        return edge_combination_cost
 

    def get_components(self, min_vertices, output_folder, voxel_size):
        print "Find connected components...\n"
        component_masks, hist = G.get_component_masks(self, min_vertices)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        plt.bar(range(len(hist)), hist)
        plt.savefig(output_folder + "cc_hist.png")
 
        print "Filter Graphs..."
        cc_list = []
        
        n = 0
        len_masks = len(component_masks)

        for mask in component_masks:
            print "Filter graph {}/{}".format(n, len_masks) 
            output_file = output_folder + "cc{}_min{}_phy.gt".format(n, min_vertices)
            cc_list.append(output_file)
            
            G.set_vertex_filter(self, mask)
            g1_masked = G1(0, G_in=self)
            g1_masked.save(output_file)
            
            
            G.set_vertex_filter(self, None)
            n += 1

        return cc_list
