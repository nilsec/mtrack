from graphs import g1_graph, graph_converter,\
                   cost_converter, g2_solver

from preprocessing import g1_to_nml, extract_candidates,\
                          DirectionType, candidates_to_g1,\
                          connect_graph_locally

from postprocessing import combine_knossos_solutions,\
                           combine_gt_solutions

from timeit import default_timer as timer
import os
import json

def solve(g1,
          start_edge_prior,
          distance_factor,
          orientation_factor,
          comb_angle_factor,
          selection_cost,
          time_limit,
          output_dir=None,
          voxel_size=None):

    """
    Base solver given a g1 graph.
    """

    vertex_cost_params = {}
    edge_cost_params = {"distance_factor": distance_factor,
                        "orientation_factor": orientation_factor,
                        "start_edge_prior": start_edge_prior}

    edge_combination_cost_params = {"comb_angle_factor": comb_angle_factor}

    

    if isinstance(g1, str):
        g1_tmp = g1_graph.G1(0) # initialize empty G1 graph
        g1_tmp.load(g1) # load from file
        g1 = g1_tmp

    if g1.get_number_of_edges() == 0:
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
        g2.set_cost(v, g2_cost[v])

    print "Create ILP..."
    solver = g2_solver.G2Solver(g2)
    
    print "Solve ILP..."
    g2_solution = solver.solve(time_limit=time_limit)

    print "Get G1 solution..."
    g1_solution = g2_to_g1_solution(g2_solution, g1, g2, index_maps)

    
    if output_dir is not None:
        assert(voxel_size is not None)

        print "Save solution..."
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        g1_to_nml(g1_solution, 
                  output_dir + "g1s_kno.nml", 
                  knossos=True, 
                  voxel_size=voxel_size)

        g1_to_nml(g1_solution, 
                  output_dir + "g1s_phy.nml")

        g1_to_nml(g1_solution, 
                  output_dir + "g1s_vox.nml", 
                  voxel=True, 
                  voxel_size=voxel_size)
        
        g1_solution.save(output_dir + "g1s.gt")

        meta_data = {"tl": time_limit,
                     "vs": voxel_size,
                     "ec": edge_cost_params,
                     "ecc": edge_combination_cost_params,
                     "sc": selection_cost}

        with open(output_dir + "meta.txt", "w+") as meta:
            json.dump(meta_data, meta)

    return g1_solution


def g2_to_g1_solution(g2_solution, g1, g2, index_maps):
    g1_selected_edges = set()
    g1_selected_vertices = set()

    for v in g2.get_vertex_iterator():
        
        if g2_solution[v] > 0.5:
            g1e_in_g2v = index_maps["g2vertex_g1edges"][v]
            
            for e in g1e_in_g2v:
                if e.source() != g1_graph.G1.START_EDGE.source():
                    g1_selected_edges.add(e)
                    g1_selected_vertices.add(e.source())
                    g1_selected_vertices.add(e.target())

    edge_mask = []
    vertex_mask = []

    g1.g.set_vertex_filter(None)
    g1.g.set_edge_filter(None)

    for v in g1.get_vertex_iterator():
        if v in g1_selected_vertices:
            vertex_mask.append(True)
        else:
            vertex_mask.append(False)

    for e in g1.get_edge_iterator():
        if e in g1_selected_edges:
            edge_mask.append(True)
        else:
            edge_mask.append(False)

    vertex_filter = g1.g.new_vertex_property("bool")
    edge_filter = g1.g.new_edge_property("bool")

    vertex_filter.a = vertex_mask
    edge_filter.a = edge_mask

    g1.g.set_vertex_filter(vertex_filter)
    g1.g.set_edge_filter(edge_filter)

    return g1


def solve_volume(volume_dir,
                 start_edge_prior,
                 distance_factor,
                 orientation_factor,
                 comb_angle_factor,
                 selection_cost,
                 time_limit,
                 output_dir,
                 voxel_size,
                 combine_solutions=True):

    """
    Solve a complete volume given connected components in .gt
    format. The folder needs to be specified in volume_dir
    """

    start = timer()

    print "Solve volume {}...".format(volume_dir)

    components = [f for f in os.listdir(volume_dir[:-1]) if f[-3:] == ".gt"]
    n_comp = len(components)

    print "with {} components...".format(n_comp)
    
    
    i = 0
    for cc in components:
        assert("phy" in cc) # assume physical coordinates

        cc_output_dir = os.path.join(os.path.join(output_dir[:-1], "ccs"),\
                                     cc[:-3]) + "/"

        print "Solve cc {}/{}".format(i + 1, n_comp)
        solve(os.path.join(volume_dir[:-1], cc),
              start_edge_prior,
              distance_factor,
              orientation_factor,
              comb_angle_factor,
              selection_cost,
              time_limit,
              cc_output_dir,
              voxel_size)
        i += 1
    
    end = timer()
    runtime = end - start

    stats = {"n_comps": n_comp,
             "runtime": runtime,
             "volume_dir": volume_dir}

    with open(output_dir + "stats.txt", "w+") as f:
        json.dump(stats, f)

    if combine_solutions:
        print "Combine Solutions...\n"
        combine_knossos_solutions(output_dir, output_dir + "volume.nml")
        combine_gt_solutions(output_dir, output_dir + "volume.gt")
        


def solve_bb_volume(bounding_box,
                    prob_map_stack,
                    gs,
                    ps,
                    distance_threshold,
                    start_edge_prior,
                    distance_factor,
                    orientation_factor,
                    comb_angle_factor,
                    selection_cost,
                    time_limit,
                    output_dir,
                    voxel_size,
                    combine_solutions=True):

    """
    Solve a volume constrained by a bounding box directly from 
    the probability maps. prob_map_stack needs to be of DirectionType
    and specify perpendicular and parallel prob map path. Bounding box
    can also only specify the section e.g. [300, 400] leads to sections
    300 to 400 are processed.
    """


    candidates = extract_candidates(prob_map_stack,
                                                  gs,
                                                  ps,
                                                  voxel_size,
                                                  bounding_box=bounding_box,
                                                  bs_output_dir=output_dir + "binary_stack/")

    g1 = candidates_to_g1(candidates, 
                          voxel_size)

    g1_connected = connect_graph_locally(g1,
                                         distance_threshold)
    
    cc_list = g1_connected.get_components(min_vertices=4,
                                          output_folder = output_dir + "cc/")

    solve_volume(output_dir + "cc/",
                 start_edge_prior,
                 distance_factor,
                 orientation_factor,
                 comb_angle_factor,
                 selection_cost,
                 time_limit,
                 output_dir + "solution/",
                 voxel_size,
                 combine_solutions)


if __name__ == "__main__":

    distance_threshold = 150
    start_edge_prior = 180.0
    distance_factor = 0.0
    orientation_factor = 15.0
    comb_angle_factor = 16.0
    selection_cost = -80.0
    output_dir = "/media/nilsec/d0/gt_mt_data/solve_volumes/test_volume_300_310/"
    time_limit = 1000
    voxel_size = [5.0, 5.0, 50.0]
    bounding_box = [300, 310]
    gs = DirectionType(0.5, 0.5)
    ps = DirectionType(0.4, 0.4)

    prob_map_stack_file_perp_test = "/media/nilsec/d0/gt_mt_data/" +\
                               "probability_maps/test/perpendicular/stack/stack.h5"
    
    prob_map_stack_file_par_test = "/media/nilsec/d0/gt_mt_data/" +\
                               "probability_maps/test/parallel/stack/stack.h5"
 
    prob_map_stack = DirectionType(prob_map_stack_file_perp_test,
                                                 prob_map_stack_file_par_test)
 
    solve_bb_volume(bounding_box,
                    prob_map_stack,
                    gs,
                    ps,
                    distance_threshold,
                    start_edge_prior,
                    distance_factor,
                    orientation_factor,
                    comb_angle_factor,
                    selection_cost,
                    time_limit,
                    output_dir,
                    voxel_size) 
