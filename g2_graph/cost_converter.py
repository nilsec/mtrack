import graph_converter

class CostConverter:
    def __init__(self, 
                 g1, 
                 vertex_cost_params,
                 edge_cost_params,
                 edge_combination_cost_params):

        self.g1 = g1
        self.vcps = vertex_cost_params
        self.ecps = edge_cost_params
        self.eccps = edge_combination_cost_params

    def get_g2_cost(self,
                    g2,
                    index_maps,
                    selection_cost,
                    start_edge_prior, 
                    vertex_cost_params, 
                    edge_cost_params, 
                    edge_combination_cost_params):

        g1_vertex_cost = self.g1.get_vertex_cost(**self.vccps)
        g1_edge_cost = self.g1.get_edge_cost(**ecps)
        g1_edge_combination_cost = self.g1.get_combination_cost(**eccps)

        g2_index_map = g2.get_edge_index_map()

        for g2_v in self.g2.get_vertex_iterator():
            g2_v_id = g2_index_map[g2_v]

            g1_edges = index_maps["g2vertex_g1edges"][g2_v_id]
            g1_vertex_center = index_maps["g1_vertex_center"][g2_v_id]

            g1_v_e1 = [g1_edges[0].source(), g1_edges[0].target()]
            g1_v_e2 = [g1_edges[1].source(), g1_edges[1].target()]

            g1_v_distinct = [v for v in g1_v_e1 + g1_v_e2 if v != g1_vertex_center]
            g1_v_distinct += [g1_vertex_center]

            g2_v_vertex_cost = 0.0
            for g1_v in g1_v_distinct:
                g2_v_vertex_cost += g1_vertex_cost[g1_v]
            
            g2_v_edge_cost = 0.0
            for g1_e in g1_edges:
                g2_v_edge_cost += g1_edge_cost[g1_e]

            g2_v_combined_edge_cost = g1_edge_combination_cost[g1_edges]

            g2_v_selection_cost = selection_cost

            NOTE: START EDGE PRIOR MISSING! LIKELY NEED TO PUT IT IN EDGE COMBINATION COST
