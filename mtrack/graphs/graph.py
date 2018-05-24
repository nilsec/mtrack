import graph_tool.all as gt
import numpy as np

class G:
    """
    Base graph class
    """
    def __init__(self, N, G_in=None):
        if G_in is None:
            self.N = N
            self.g = gt.Graph(directed=False)
            self.g.add_vertex(N)
        else:
            g = G_in.g.copy()
            self.g = gt.Graph(g, prune=False, directed=False)

    def get_shortest_path(self, source, target, weights):
        vertex_list, edge_list = gt.shortest_path(self.g, source, target, weights)
        return vertex_list, edge_list

    def get_number_of_edges(self):
        return self.g.num_edges()

    def get_number_of_vertices(self):
        return self.g.num_vertices()

    def get_vertex(self, u):
        return self.g.vertex(u, use_index=True, add_missing=False)

    def get_edge(self, u, v):
        edges = self.g.edge(u,v, all_edges=True, add_missing=False)
        # Return all edges but this list should never contain
        # more then one edge. Can be empty if the requested edge
        # is not in the graph.
        assert(len(edges) <= 1)

        try:
            return edges[0]
        except IndexError:
            raise ValueError("Nonexistent edge: ({},{})".format(u, v))

    def add_vertex(self):
        """
        Add vertex and return 
        vertex descriptor.
        """
        u = self.g.add_vertex()
        return u

    def add_edge(self, u, v, reindex=False):
        """
        Add edge and return 
        edge descriptor that has
        methods .source()
        .target(). Get the id
        of the edge from 
        edge_index_map[e]
        """
        # Check that vertices exist:
        u_there = self.get_vertex(u)
        v_there = self.get_vertex(v)

        # Check that they are distinct - no self edges allowed:
        assert(u_there != v_there)

        # Check that edge does not exist
        edges = self.g.edge(u,v, all_edges=True, add_missing=False)
        assert(len(edges) == 0)
        
        # Add edge if it does not exist
        e = self.g.add_edge(u, v)

        return e
 
    def get_vertex_index_map(self):
        return self.g.vertex_index

    def get_edge_index_map(self):
        return self.g.edge_index

    def get_vertex_iterator(self):
        """
        SLOW
        """
        return self.g.vertices()

    def get_edge_iterator(self):
        """
        SLOW if used to iterate over edges
        because of python loop. Order of edges
        DOES NOT neceressarily correspond to
        the edge index ordering as given by the
        edge_index property map.
        """
        return self.g.edges()

    def get_vertex_array(self):
        return self.g.get_vertices()
 
    def get_edge_array(self):
        """
        graph_tool.get_edges()
        Return a numpy.ndarray containing the edges. The shape of
        the array will be (E, 3) where E is the number of edges and
        each line will contain the source, target and index of an edge.
        """
        return self.g.get_edges()

    def new_vertex_property(self, name, dtype, set_vp=True, value=None, array=False):
        if value is None:
            vp = self.g.new_vertex_property(dtype)
        else:
            if array:
                vp = self.g.new_vertex_property(dtype)
                vp.set_2d_array(value)
            else:
                vp = self.g.new_vertex_property(dtype, vals=value)

        if set_vp:
            self.g.vertex_properties[name] = vp

        return vp

    def set_vertex_property(self, name, u, value, vals=None):
        """
        NOTE: For vector types an empty 
        array is initialized.
        For double, int etc. a zero is set as
        default value.
        """
        if vals is None:
            if not isinstance(u, gt.Vertex):
                if isinstance(u, int):
                    u = self.get_vertex(u)
                else:
                    raise ValueError("Vertex has to be int or gt.Vertex")

            self.g.vertex_properties[name][u] = value
        else:
            self.g.vertex_properties[name].a = vals

    def get_vertex_property(self, name, u=None):
        if u is not None:
            u = self.get_vertex(u)
            return self.g.vertex_properties[name][u]
        else:
            return self.g.vertex_properties[name]

    def new_edge_property(self, name, dtype, vals=None):
        ep = self.g.new_edge_property(dtype, vals=vals)
        self.g.edge_properties[name] = ep
        return ep

    def set_edge_property(self, name, u, v, value, e=None):
        if e is None:
            e = self.get_edge(u, v)
        self.g.edge_properties[name][e] = value
    
    def get_edge_property(self, name, u=None, v=None):
        if (u is not None) and (v is not None):
            e = self.get_edge(u, v)
            return self.g.edge_properties[name][e]
        elif (u is None) and (v is None):
            return self.g.edge_properties[name]
        else:
            raise ValueError("Provide both u AND v or nothing.")

    def new_graph_property(self, name, dtype):
        gp = self.g.new_graph_property(dtype)
        self.g.graph_properties[name] = gp
        return gp

    def set_graph_property(self, name, value):
        self.g.graph_properties[name] = value

    def get_graph_property(self, name):
        return self.g.graph_properties[name] 

    def list_properties(self):
        self.g.list_properties()
  
    def get_neighbour_nodes(self, u):
        """
        Returns sorted vertex id's.
        """
        return list(np.sort(self.g.get_out_neighbours(u)))

    def get_incident_edges(self, u):
        """
        Returns incident edges sorted by id.
        """
        edges = self.g.get_out_edges(u)
        edges = np.array(sorted(edges, key=lambda x: x[2]))

        return [self.get_edge(e[0], e[1]) for e in edges]

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

    def set_vertex_filter(self, vertex_property):
        self.g.set_vertex_filter(vertex_property)

    def set_vertex_mask(self, vertex_mask):
        vp = self.g.new_vertex_property("bool")
        vp.a = vertex_mask
        self.set_vertex_filter(vp)

    def get_articulation_points(self):
        bicomp_ep, articulation_vp, nc = gt.label_biconnected_components(self.g)
        naps = articulation_vp.a == 0
        naps_vp = self.g.new_vertex_property("bool")
        naps_vp.a = naps
        return naps_vp

    def get_nested_sbm_masks(self, edge_weights, rec_types="real-exponential"):
        if edge_weights:
            print "Minimize bested block model with edge weights..."
            state_args = dict(recs=[self.g.ep.weight], rec_types=[rec_types])
        else:
            print "Minimize nested block model..."
            state_args = {}
        
        state = gt.minimize_nested_blockmodel_dl(self.g, state_args=dict(recs=[self.g.ep.weight], 
                                                     rec_types=[rec_types]))
        states = state.get_levels()

        print "Generate masks..."
        mask_list = []
        for state in states:
            max_comp = state.B
            masks = []
            for label in range(0, max_comp):
                print state.b.a
                print len(state.b.a)
                binary_mask = state.b.a == label
            
                cc_vp = self.g.new_vertex_property("bool")
                try:
                    cc_vp.a = binary_mask
                    masks.append(cc_vp)
                except:
                    continue

            mask_list.append(masks)

        return mask_list
 
 
    def get_sbm_masks(self):
        print "Minimize block model..."
        state = gt.minimize_blockmodel_dl(self.g)
        
        print "Generate masks..."
        max_comp = state.B
        masks = []
        for label in range(0, max_comp):
            print state.b.a
            print len(state.b.a)
            binary_mask = state.b.a == label
            
            cc_vp = self.g.new_vertex_property("bool")
            cc_vp.a = binary_mask
            masks.append(cc_vp)
        return masks

    def get_kcore_mask(self, min_k):
        kval_vp = gt.kcore_decomposition(self.g)
        mask = kval_vp.a >= min_k
        mask_vp = self.g.new_vertex_property("bool")
        mask_vp.a = mask
        return mask_vp

    def get_min_cut_masks(self):
        print "Get min cut masks..."
        min_cut, partition = gt.min_cut(self.g, self.g.ep.weight)
        print "min cut: ", min_cut
 
        masks = []
        for p in [0,1]:
            mask_vp = self.g.new_vertex_property("bool")
            mask_vp.a = partition.a == p
            masks.append(mask_vp)

        return masks
 
    def get_component_masks(self, min_vertices=0):
        component_vp, hist = gt.label_components(self.g, 
                                                 directed=False,
                                                 attractors=False)

        print "Generate Masks..."
        masks = []
        max_comp = max(component_vp.a)
        for label in range(0, max_comp):
            if hist[label] < min_vertices:
                continue
  
            print "Get label {}/{}".format(label, max_comp)
            binary_mask = component_vp.a == label

            cc_vp = self.g.new_vertex_property("bool")
            cc_vp.a = binary_mask
            masks.append(cc_vp)

        return masks, hist

    def save(self, path):
        self.g.save(path, fmt='gt')

    def load(self, path):
        self.g.load(path, fmt='gt')
