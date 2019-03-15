

class CostConverter:
    def __init__(self, 
                 g1, 
                 vertex_cost_params,
                 edge_cost_params,
                 edge_combination_cost_params,
                 selection_cost,
                 edge_combination_cost="angle"):

        self.g1 = g1
        self.vcps = vertex_cost_params
        self.ecps = edge_cost_params
        self.eccps = edge_combination_cost_params
        self.selection_cost = selection_cost
        if not edge_combination_cost in ["angle", "curvature"]:
            raise ValueError("Choose between /angle/ or /curvature/")
        self.edge_combination_cost = edge_combination_cost

    def get_g2_cost(self,
                    g2,
                    index_maps):

        g1_vertex_cost = self.g1.get_vertex_cost(**self.vcps)
        g1_edge_cost = self.g1.get_edge_costs(**self.ecps)
        if self.edge_combination_cost == "angle":
            g1_edge_combination_cost = self.g1.get_edge_combination_cost_angle(**self.eccps)
        else:
            g1_edge_combination_cost = self.g1.get_edge_combination_cost_curvature(**self.eccps)

        g2_index_map = g2.get_vertex_index_map()
        g2_cost = {}

        for g2_v in g2.get_vertex_iterator():
            g2_v_id = g2_index_map[g2_v]

            g1_edges = index_maps["g2vertex_g1edges"][g2_v_id]
            g1_vertex_center = index_maps["g1_vertex_center"][g2_v_id]

            g1_v_e1 = [g1_edges[0].source(), g1_edges[0].target()]
            g1_v_e2 = [g1_edges[1].source(), g1_edges[1].target()]

            g1_v_distinct = [v for v in g1_v_e1 + g1_v_e2 if v != g1_vertex_center]
            g1_v_distinct += [g1_vertex_center]

            g2_v_vertex_cost = 0.0
            for g1_v in g1_v_distinct:
                # 3 vertices in one g2 vertex: Scale with 1/3
                g2_v_vertex_cost += (1./3) * g1_vertex_cost[g1_v]
            
            g2_v_edge_cost = 0.0
            for g1_e in g1_edges:
                # 2 edges in one g2 vertex: Scale with 1/2
                g2_v_edge_cost += (1./2) * g1_edge_cost[g1_e]

            # No scaling for combined edges, 1 to 1 correspondence
            g2_v_combined_edge_cost = (1.) * g1_edge_combination_cost[g1_edges]

            g2_v_selection_cost = self.selection_cost
            
            g2_cost[g2_v] = g2_v_vertex_cost + g2_v_edge_cost +\
                            g2_v_combined_edge_cost + g2_v_selection_cost

        return g2_cost
