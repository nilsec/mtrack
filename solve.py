from graphs import g1_graph
from graphs import graph_converter
from graphs import cost_converter
from graphs import g2_solver

def solve(g1,
          start_edge_prior,
          distance_factor,
          orientation_factor,
          comb_angle_factor,
          selection_cost,
          output_directory,
          time_limit)

    vertex_cost_params = {}
    edge_cost_params = {"distance_factor": distance_factor,
                        "orientation_factor": orientation_factor,
                        "start_edge_prior": start_edge_prior}

    edge_combination_cost_params = {"comb_angle_factor": 1.0}

    if isinstance(g1, str):
        g1_tmp = g1_graph.G1(0) # initialize empty G1 graph
        g1 = g1_tmp.load(g1) # load from file

    if self.g1.get_number_of_edges() == 0:
        raise Warning("Graph has no edges.")

    print "Get G2 graph..."
    g_converter = graph_converter.GraphConverter(g1)
    g2, index_maps = g_converter.get_g2_graph()

    print "Get G2 costs..."
    c_converter = cost_converter.CostConverter(g1,
                                               vertex_cost_params,
                                               edge_cost_params,
                                               edge_combination_cost_params,
                                               selection_cost)
    g2_cost = c_converter.get_g2_cost(g2, index_maps)

    print "Set G2 costs..."
    for v in g2.get_vertex_iterator():
        self.g2.set_cost(v, g2_cost[v])

    print "Create ILP..."
    solver = g2_solver.G2Solver(g2)
    
    print "Solve ILP..."
    g2_solution = solver.solve(time_limit=time_limit, verbose=True)

    return g2_solution    
