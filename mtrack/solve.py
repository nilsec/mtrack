import numpy as np
import os
import json
import logging

from mtrack.graphs import g1_graph, graph_converter,\
                   cost_converter, g2_solver

def solve(g1,
          start_edge_prior,
          orientation_factor,
          comb_angle_factor,
          selection_cost,
          time_limit,
          output_dir=None,
          voxel_size=None,
          chunk_shift=np.array([0.,0.,0.]),
          backend="Gurobi"):

    """
    Base solver given a g1 graph.
    """

    vertex_cost_params = {}
    edge_cost_params = {"orientation_factor": orientation_factor,
                        "start_edge_prior": start_edge_prior}

    edge_combination_cost_params = {"comb_angle_factor": comb_angle_factor}

    

    if isinstance(g1, str):
        g1_tmp = g1_graph.G1(0) # initialize empty G1 graph
        g1_tmp.load(g1) # load from file
        g1 = g1_tmp

    if g1.get_number_of_edges() == 0:
        raise Warning("Graph has no edges.")


    logging.info("Get G2 graph...")
    g_converter = graph_converter.GraphConverter(g1)
    g2, index_maps = g_converter.get_g2_graph()

    logging.info("Get G2 costs...")
    c_converter = cost_converter.CostConverter(g1,
                                               vertex_cost_params,
                                               edge_cost_params,
                                               edge_combination_cost_params,
                                               selection_cost)
    g2_cost = c_converter.get_g2_cost(g2, index_maps)

    logging.info("Set G2 costs...")
    for v in g2.get_vertex_iterator():
        g2.set_cost(v, g2_cost[v])

    logging.info("Create ILP...")
    solver = g2_solver.G2Solver(g2, backend=backend)
    
    logging.info("Solve ILP...")
    g2_solution = solver.solve(time_limit=time_limit)

    if len(g2_solution) != solver.g2_vertices_N:
        raise ValueError("Model infeasible")

    logging.info("Get G1 solution...")
    g1_solution = g2_to_g1_solution(g2_solution, 
                                    g1, 
                                    g2, 
                                    index_maps, 
                                    chunk_shift=chunk_shift)
   

    if output_dir is not None:
        assert(voxel_size is not None)

        logging.info("Save solution...")
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

        with open(output_dir + "meta.json", "w+") as meta:
            json.dump(meta_data, meta)

    return g1_solution


def g2_to_g1_solution(g2_solution, 
                      g1, 
                      g2, 
                      index_maps, 
                      voxel_size=[5.,5.,50.], 
                      chunk_shift=np.array([0.,0.,0.])):

    g1_selected_edges = set()
    g1_selected_vertices = set()

    for v in g2.get_vertex_iterator():
        
        if g2_solution[int(v)] > 0.5:
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
            pos = g1.get_position(v)
            pos += np.array(chunk_shift) * np.array(voxel_size)
            g1.set_position(v, np.array(pos))

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
