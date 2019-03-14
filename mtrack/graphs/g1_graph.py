import numpy as np
from numpy.core.umath_tests import inner1d
import os
import itertools
import logging

from mtrack.graphs.graph import G
from mtrack.graphs.start_edge import StartEdge 
from mtrack.mt_utils.spline_interpolation import get_energy_from_ordered_points

class G1(G):
    START_EDGE = StartEdge()
    
    def __init__(self, N, G_in=None, init_empty=False):
        G.__init__(self, N, G_in=G_in)
        self.hash_value = None

        if not init_empty:
            if G_in is None:
                G.new_vertex_property(self, "orientation", dtype="vector<double>")
                G.new_vertex_property(self, "position", dtype="vector<double>")
                G.new_vertex_property(self, "partner", dtype="long long")

                G.new_vertex_property(self, "selected", dtype="bool", value=False)
                G.new_vertex_property(self, "solved", dtype="bool", value=False)
                G.new_edge_property(self, "selected", dtype="bool", value=False)
                G.new_edge_property(self, "solved", dtype="bool", value=False)
                G.new_edge_property(self, "edge_cost", dtype="double", value=0.0)
        
                for v in G.get_vertex_iterator(self):
                    self.set_partner(v, -1) 

        # NOTE: Start/End dummy node and edge are implicit
        self.edge_index_map = G.get_edge_index_map(self)

    def set_hash(self, value):
        self.hash_value = value

    def __hash__(self):
        if self.hash_value is None:
            raise TypeError("Object is not hashable.")
        else:
            return self.hash_value

    def __eq__(self, other):
        # Number of vertices:
        n_vertices = (self.get_number_of_vertices() == other.get_number_of_vertices())
        if not n_vertices:
            return False

        # Number of edges:
        n_edges = (self.get_number_of_edges() == other.get_number_of_edges())
        if not n_edges:
            return False

        # Vertex comparison
        other_to_self = {}
        attr_other = []
        attr_self = []

        for v_self in self.get_vertex_iterator():
            pos_self = self.get_position(v_self)
            ori_self = self.get_orientation(v_self)
            par_self = self.get_partner(v_self)
            attr_self.append(tuple(list(pos_self) + list(ori_self) + [par_self]))

            n_matches = 0
            for v_other in other.get_vertex_iterator():
                pos_other = other.get_position(v_other)
                ori_other = other.get_orientation(v_other)
                par_other = other.get_partner(v_other)
                
                if np.allclose(pos_self, pos_other):
                    if np.allclose(ori_self, ori_other):
                        other_to_self[v_other] = v_self
                        n_matches += 1

            if n_matches == 0:
                return False

            if n_matches > 1:
                raise ValueError("Double match found in equality")

        for v_other in other.get_vertex_iterator():
            pos_other = other.get_position(v_other)
            ori_other = other.get_orientation(v_other)
            par_other = other.get_partner(v_other)
            attr_other.append(tuple(list(pos_other) + list(ori_other) + [par_other]))

        if len(attr_self) != len(set(attr_self)):
            raise ValueError("LHS graph not valid")
        if len(attr_other) != len(set(attr_other)):
            raise ValueError("RHS graph not valid")

        # Edge comparison
        for v_other, v_self in other_to_self.iteritems():
            incident_self = self.get_incident_edges(v_self)
            incident_other = other.get_incident_edges(v_other)

            if len(incident_self) != len(incident_other):
                return False
            
            vedges_self = []
            for e in incident_self:
                v0 = e.source()
                v1 = e.target()
                vedges_self.append(tuple(sorted([v0,v1])))

            vedges_other = []
            for e in incident_other:
                v0 = other_to_self[e.source()]
                v1 = other_to_self[e.target()]
                vedges_other.append(tuple(sorted([v0, v1])))

            for vedge in vedges_self:
                if not (vedge in vedges_other):
                    return False
            
            for vedge in vedges_other:
                if not (vedge in vedges_self):
                    return False

        return True 
                    
        
    def add_edge(self, u, v):
        e = G.add_edge(self, u, v)
        return e

    def add_edge_list(self, edges, hashed=False):
        self.g.add_edge_list(edges, hashed)

    def select_vertex(self, u):
        G.set_vertex_property(self, "selected", u, True)

    def solve_vertex(self, u):
        G.set_vertex_property(self, "solved", u, True)

    def select_edge(self, e):
        G.set_edge_property(self, "selected", None, None, True, e)

    def solve_edge(self, e):
        G.set_edge_property(self, "solved", None, None, True, e)

    def set_edge_cost(self, e, edge_cost):
        G.set_edge_property(self, "edge_cost", None, None, edge_cost, e)

    def get_edge_cost(self, e):
        return G.get_edge_property(self, "edge_cost", e=e)
        
    def set_orientation(self, u, value):
        assert(type(value) == np.ndarray)
        assert(len(value) == 3)
        assert(value.ndim == 1)

        G.set_vertex_property(self, "orientation", u, value)
        
    def get_orientation(self, u):
        orientations = G.get_vertex_property(self, "orientation")
        return orientations[u]


    def get_orientation_array(self):
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

    def set_partner(self, u, value, vals=None):
        assert(type(value) == int)
        assert(value < G.get_number_of_vertices(self))
        
        if vals is None:
            G.set_vertex_property(self, "partner", u, value)
        else:
            G.set_vertex_property(self, "partner", 0, 0, vals=vals)
        
    def get_partner(self, u):
        partner = G.get_vertex_property(self, "partner")
        return partner[u]

    def get_partner_array(self):
        partner = G.get_vertex_property(self, "partner")
        return partner.get_array()

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

    def reindex_edges_save(self):
        """
        Reindex edges to range 0,N
        but preserve the selected & solved edge property
        by saving the (unchanged) vertex 
        indices of edges before reindexing.
        """ 
        
        selected_ep = self.get_edge_property("selected")
        solved_ep = self.get_edge_property("solved")        

        """
        Save vertex ids
        """
        selected_edges = []
        solved_edges = []
        for e in self.get_edge_iterator():
            if selected_ep[e]:
                selected_edges.append((e.source(), e.target()))
                assert(solved_ep[e])
                solved_edges.append((e.source(), e.target()))

            elif solved_ep[e]:
                solved_edges.append((e.source(), e.target()))
        
        # Reindex edges
        self.g.reindex_edges()

        """
        Initialize new property maps with old
        vertex ids.
        """
        new_selected_edges = self.g.new_edge_property("bool")
        for e in selected_edges:
            new_selected_edges[self.get_edge(*e)] = True

        new_solved_edges = self.g.new_edge_property("bool")
        for e in solved_edges:
            new_solved_edges[self.get_edge(*e)] = True
         
        self.g.edge_properties["selected"] = new_selected_edges
        self.g.edge_properties["solved"] = new_solved_edges
                
                
    def get_edge_costs(self, orientation_factor, start_edge_prior):
        edge_cost_tot = {e: self.get_edge_cost(e) * orientation_factor for e in G.get_edge_iterator(self)}
        edge_cost_tot[self.START_EDGE] = 2.0 * start_edge_prior
        return edge_cost_tot


    def get_edge_combination_cost(self, 
                                  comb_angle_factor, 
                                  comb_angle_prior=0.0,
                                  return_edges_to_middle=False):

        edge_index_map = G.get_edge_index_map(self)
        """
        Only collect the indices for each edge combination in
        the loop and perform cost calculation later in vectorized 
        form.
        """
        middle_indices = []
        edges_to_middle = {}
        end_indices = []
        edges = []
        cost = []
        prior_cost = {}
        edge_combination_cost = {}
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
                        edges_to_middle[(e1, e2)] = int(v)
 
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

                        end_vertices = [int(e1.source()), 
                                        int(e1.target()), 
                                        int(e2.source()), 
                                        int(e2.target())]

                        end_vertices.remove(middle_vertex)
                        end_vertices.remove(middle_vertex)

                        ordered_points = [None, None, None]
                        ordered_points[0] = np.array(self.get_position(end_vertices[0]))
                        ordered_points[2] = np.array(self.get_position(end_vertices[1]))
                        ordered_points[1] = np.array(self.get_position(middle_vertex))

                        energy = get_energy_from_ordered_points(ordered_points, n_samples=1000)
                        edge_combination_cost[(e1, e2)] = (energy * comb_angle_factor)**2


                        end_indices.extend(list(end_vertices))
                        edges.append((e1, e2))
                        edges_to_middle[(e1, e2)] = int(v)
                        """
                        (e1, e2) -> end_indices, middle_indices
                        """

        edge_combination_cost.update(prior_cost)
        logging.info("edge_combination_cost: " + str(edge_combination_cost))

        if return_edges_to_middle:
            return edge_combination_cost, edges_to_middle
        else:
            return edge_combination_cost
    
    def get_sbm(self, output_folder, nested=False, edge_weights=False):
        logging.info("Find SBM partition...")
        if nested:
            mask_list = G.get_nested_sbm_masks(self, edge_weights)
        else:
            mask_list = [G.get_sbm_masks(self)]

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        logging.info("Filter Graphs...")
        cc_path_list = []
        
        n = 0
        l = 0
        levels = len(mask_list)
        level_path_list = []
        for masks in mask_list:
            for mask in masks:
                output_file = output_folder.replace("/", "level_%s/" % l) + "cc{}.gt".format(n)
                cc_path_list.append(output_file)
            
                G.set_vertex_filter(self, mask)
                g1_masked = G1(0, G_in=self)
                if not os.path.exists(os.path.dirname(output_file)):
                    os.makedirs(os.path.dirname(output_file))
                g1_masked.save(output_file)
            
                G.set_vertex_filter(self, None)
                n += 1
            level_path_list.append(cc_path_list)
            l += 1

        return level_path_list
    
    def get_min_cut(self, g):
        masks = G.get_min_cut_masks(g)
        partition = []

        partition_vertices = []
        for mask in masks:
            G.set_vertex_filter(g, mask)
            partition_vertices.append(g.get_number_of_vertices())
            g1_masked = G1(0, G_in=g)
            g1_masked.g.purge_vertices()
            partition.append(g1_masked)
            G.set_vertex_filter(g, None)
    
        small_partition_idx = sorted(enumerate(partition_vertices), key= lambda x: x[1])[0][0]
        small = partition[small_partition_idx] 
        big = partition[int(not small_partition_idx)]

        cut_edges = g.get_number_of_edges() - big.get_number_of_edges() - small.get_number_of_edges()

        return partition, cut_edges

    def get_hcs(self, g, remove_singletons=1, hcs=[]):
        logging.info("Get hcs...")
        if remove_singletons:
            logging.info("Remove Singletons")
            singleton_mask = G.get_kcore_mask(g, remove_singletons)
            g.g.set_vertex_filter(singleton_mask)
            g.g.purge_vertices()
             
        partition, cut_edges = self.get_min_cut(g)
        
        if cut_edges > g.get_number_of_vertices()/2.:
            hcs.append(g)
        else:
            if partition[0].get_number_of_vertices()>1:
                self.get_hcs(g=partition[0], remove_singletons=False, hcs=hcs)

            if partition[1].get_number_of_vertices()>1:
                self.get_hcs(g=partition[1], remove_singletons=False, hcs=hcs)

        return hcs
        
    def get_components(self, 
                       min_vertices, 
                       output_folder, 
                       remove_aps=False, 
                       min_k=1,
                       return_graphs=False):

        logging.info("Get components...")
        if remove_aps:
            logging.info("Remove articulation points...")
            naps_vp = G.get_articulation_points(self)
            G.set_vertex_filter(self, naps_vp)

        if min_k > 1:
            logging.info("Find " + str(min_k) + "-cores...")
 
            kcore_vp = G.get_kcore_mask(self, min_k)
            G.set_vertex_filter(self, kcore_vp)

        logging.info("Find connected components...")
        masks, hist = G.get_component_masks(self, min_vertices)
        
        if output_folder is not None:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
    
        logging.info("Filter Graphs...")
        cc_path_list = []
       
        graph_list = [] 
        n = 0
        len_masks = len(masks)
        for mask in masks:
            logging.info("Filter graph " + str(n) + "/" + str(len_masks)) 
            if output_folder is not None:
                output_file = output_folder +\
                                "cc{}_min{}_phy.gt".format(n, min_vertices)
                cc_path_list.append(output_file)
           
            
            G.set_vertex_filter(self, mask)
            g1_masked = G1(0)
            g1_masked.g = self.g.copy()
            graph_list.append(g1_masked)
            
            if output_folder is not None:
                g1_masked.save(output_file)
            
            G.set_vertex_filter(self, None)
            n += 1
        if return_graphs:
            return graph_list
        else:
            return cc_path_list
