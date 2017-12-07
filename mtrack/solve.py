import numpy as np
import os
import json
from timeit import default_timer as timer
from shutil import copyfile
from copy import deepcopy
import glob
import h5py
from pymongo import MongoClient, IndexModel, ASCENDING
from scipy.spatial import KDTree
import pdb


from mtrack.graphs import g1_graph, graph_converter,\
                   cost_converter, g2_solver

from mtrack.preprocessing import g1_to_nml, extract_candidates,\
                          DirectionType, candidates_to_g1,\
                          connect_graph_locally, Chunker,\
                          slices_to_chunks, Chunk

from mtrack.postprocessing import combine_knossos_solutions,\
                           combine_gt_solutions



def solve(g1,
          start_edge_prior,
          distance_factor,
          orientation_factor,
          comb_angle_factor,
          selection_cost,
          time_limit,
          output_dir=None,
          voxel_size=None,
          z_correction=None,
          chunk_shift=np.array([0.,0.,0.])):

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
    g1_solution = g2_to_g1_solution(g2_solution, 
                                    g1, 
                                    g2, 
                                    index_maps, 
                                    z_correction=z_correction, 
                                    chunk_shift=chunk_shift)
   
    """ 
    print "Warning, purging vertices"
    g1_solution.g.purge_vertices()
    g1_solution.g.purge_edges()
    """    

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

        """
        print "Warning, purging vertices"
        g1_solution.g.purge_vertices()
        g1_solution.g.purge_edges()
        """
 
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
                      z_correction=None, 
                      chunk_shift=np.array([0.,0.,0.])):

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
            # Revert the z-position to lie on the section:
            z_factor = 0.5

            if z_correction is not None:
                z_factor += 1

            z_shift = z_factor * voxel_size[2]
            pos = g1.get_position(v)
            pos[2] += z_shift
            pos += chunk_shift * np.array(voxel_size)
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


def solve_volume(volume_dir,
                 start_edge_prior,
                 distance_factor,
                 orientation_factor,
                 comb_angle_factor,
                 selection_cost,
                 time_limit,
                 output_dir,
                 voxel_size,
                 combine_solutions=True,
                 z_correction=None,
                 chunk_shift=np.array([0.,0.,0.])):

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
        g1_solution = solve(os.path.join(volume_dir[:-1], cc),
              start_edge_prior,
              distance_factor,
              orientation_factor,
              comb_angle_factor,
              selection_cost,
              time_limit,
              cc_output_dir,
              voxel_size,
              z_correction=z_correction,
              chunk_shift=chunk_shift)

        if i == 0:
            copyfile(cc_output_dir + "meta.json", output_dir + "solve_params.json")
            
        i += 1
    
    end = timer()
    runtime = end - start

    stats = {"n_comps": n_comp,
             "runtime": runtime,
             "volume_dir": volume_dir}

    if components:
        with open(output_dir + "solve_stats.json", "w+") as f:
            json.dump(stats, f)

        if combine_solutions:
            print "Combine Solutions...\n"
            combine_knossos_solutions(output_dir, output_dir + "volume.nml")
            combined_graph = combine_gt_solutions(output_dir, output_dir + "volume.gt", purge=True)

            return combined_graph


class CoreSolver(object):
    def __init__(self):
        self.template_vertex = {"px": None,
                                "py": None,
                                "pz": None,
                                "ox": None,
                                "oy": None,
                                "oz": None,
                                "id_partner_global": None,
                                "id_global": None}


        self.template_edge = {"id_v0_global": None,
                              "id_v1_global": None,
                              "id_v0_mongo": None,
                              "id_v1_mongo": None}


    def _get_client(self, name_db, collection="graph", overwrite=False):
        print "Get client..."

        client = MongoClient()

        db = client[name_db]
        collections = db.collection_names()
        
        if collection in collections:
            if overwrite:
                print "Warning, overwrite collection!"
                self.create_collection(name_db=name_db, 
                                       collection=collection, 
                                       overwrite=True)

            print "Collection already exists, request {}.{}...".format(name_db, collection)
        else:
            print "Database empty, create...".format(name_db, collection)
            self.create_collection(name_db=name_db, 
                                   collection=collection, 
                                   overwrite=False)
            
        graph = db[collection]
        
        return graph


    def create_collection(self, name_db, collection="graph", overwrite=False):
        print "Create new db collection {}.{}...".format(name_db, collection)
        client = MongoClient()
        
        if overwrite:
            print "Overwrite {}.{}...".format(name_db, collection)
            client.drop_database(name_db)

        db = client[name_db]
        graph = db[collection]

        print "Generate indices..."
        graph.create_index([("pz", ASCENDING), ("py", ASCENDING), ("px", ASCENDING)], 
                           name="pos", 
                           sparse=True)

        graph.create_index([("id_v0_mongo", ASCENDING), ("id_v1_mongo", ASCENDING)], 
                           name="vedge_id", 
                           sparse=True)


    def save_g1_graph(self, g1_graph, name_db, collection="graph", overwrite=False):
        """
        db/gt 
        """
        graph = self._get_client(name_db, collection, overwrite)

        vertex_positions = g1_graph.get_position_array().T
        vertex_orientations = g1_graph.get_orientation_array().T
        partner = g1_graph.get_vertex_property("partner").a
        edges = g1_graph.get_edge_array()

        print "Insert vertices..."

        index_map = {}
        vertices = []
        vertex_id = 0
        for pos, ori, partner in zip(vertex_positions,
                                     vertex_orientations,
                                     partner):

            vertex = deepcopy(self.template_vertex)

            vertex["px"] = pos[0]
            vertex["py"] = pos[1]
            vertex["pz"] = pos[2]
            vertex["ox"] = ori[0]
            vertex["oy"] = ori[1]
            vertex["oz"] = ori[2]
            vertex["id_partner_global"] = partner
            vertex["id_global"] = vertex_id
        
            id_mongo = graph.insert_one(vertex).inserted_id
            index_map[vertex_id] = id_mongo 
            vertex_id += 1

        print "Insert edges..."

        for edge_id in edges:
            edge_mongo = [index_map[edge_id[0]], index_map[edge_id[1]]]
            edge = deepcopy(self.template_edge)
            
            edge["id_v0_global"] = str(edge_id[0]) # convert uint64 to str, faster & mdb doesn't accept 64 bit int
            edge["id_v1_global"] = str(edge_id[1])
            edge["id_v0_mongo"] = edge_mongo[0]
            edge["id_v1_mongo"] = edge_mongo[1]

            graph.insert_one(edge)


    def get_subgraph(self,
                     name_db,
                     collection,
                     x_lim,
                     y_lim,
                     z_lim):

        print "Extract subgraph..."
        
        graph = self._get_client(name_db, collection, overwrite=False)

        print "Perform vertex query..."

        vertices =  list(graph.find({   
                                         "pz": {"$gte": z_lim["min"],
                                                 "$lt": z_lim["max"]},
                                         "py": {"$gte": y_lim["min"],
                                                 "$lt": y_lim["max"]},
                                         "px": {"$gte": x_lim["min"],
                                                 "$lt": x_lim["max"]}
                                    
                                   })
                         )

        vertex_ids = [v["_id"] for v in vertices]
        
         

        print "Perform edge query..."
        edges = []
        for vertex in vertices:
            id_mongo = vertex["_id"]

            """
            Query all edges that contain the vertex
            and do not connect to a vertex outside
            the requested volume.
            """

            vedges = list(graph.find({"$and":
                                            [
                                                {"$or": 
                                                    [
                                                        {"id_v0_mongo": id_mongo},
                                                        {"id_v1_mongo": id_mongo}

                                                    ]
                                                },
                                                {
                                                        "id_v0_mongo": {"$in": vertex_ids},
                                                        "id_v1_mongo": {"$in": vertex_ids}
                                                }
                                            ]
                                    })
                        )

            edges.extend(vedges)
        
        print "...Done"

        if not vertices:
            print "Warning, requested region holds no vertices!"
        if not edges:
            print "Warning, requested region holds no edges!"

        return vertices, edges
        

    def subgraph_to_g1(self, vertices, edges):
        if not vertices:
            raise ValueError("Vertex list is empty")

        g1 = g1_graph.G1(len(vertices), init_empty=False)
        index_map = {}
        index_map_inv = {}
        
        partner = []
        n = 0
        for v in vertices:
            g1.set_position(n, np.array([v["px"], v["py"], v["pz"]]))
            g1.set_orientation(n, np.array([v["ox"], v["oy"], v["oz"]]))
            partner.append(v["id_partner_global"])
            
            index_map[v["id_global"]] = n
            index_map_inv[n] = [v["id_global"], v["_id"]]
            n += 1

        index_map[-1] = -1 

        index_map_get = np.vectorize(index_map.get)        
        partner = np.array(partner)
        g1.set_partner(0,0, vals=index_map_get(partner))

        n = 0
        for e in edges:
            e0 = index_map[np.uint64(e["id_v0_global"])]
            e1 = index_map[np.uint64(e["id_v1_global"])]
            g1.add_edge(e0, e1)

        return g1, index_map_inv


    def solve_subgraph(self, 
                       subgraph,
                       index_map,
                       distance_threshold,
                       cc_min_vertices,
                       start_edge_prior,
                       selection_cost,
                       distance_factor,
                       orientation_factor,
                       comb_angle_factor,
                       time_limit,
                       write=True):

        print "Solve subgraph..."

        print "Connect locally..."
        positions = subgraph.get_position_array().T
        partner = subgraph.get_partner_array()

        vertex_degrees = np.array(subgraph.g.degree_property_map("total").a)
        vertex_mask_0 = vertex_degrees == 0
        vertex_mask_1 = vertex_degrees <= 1

        index_map_0 = {sum(vertex_mask_0[:i]):i for i in range(len(vertex_mask_0)) if vertex_mask_0[i]}
        index_map_1 = {sum(vertex_mask_1[:j]):j for j in range(len(vertex_mask_1)) if vertex_mask_1[j]}
        
        positions_0 = positions[vertex_mask_0]
        positions_1 = positions[vertex_mask_1]
    
        print "Build kdtree..."
        kdtree_0 = KDTree(positions_0)
        kdtree_1 = KDTree(positions_1)

        print "Query ball tree with r={}...".format(distance_threshold)
        pairs = kdtree_0.query_ball_tree(kdtree_1, 
                                         r=distance_threshold,
                                         p=2.0)

        """
        pairs: list of lists
        For each element positions_0[i] of this tree, pairs[i] 
        is a list of the indices of its neighbors in positions_1.
        """
        index_map_1_get = np.vectorize(index_map_1.get)

        print "Add edges to subgraph..."
        edge_list = []
        idx_0 = 0
        for pair in pairs:
            idx_1_global = index_map_1_get(pair)
            idx_0_global = [index_map_0[idx_0]]
            
            edges = zip(idx_0_global * len(idx_1_global), 
                        idx_1_global)

            edges = [tuple(sorted(i)) for i in edges if (i[0] != i[1]) and (partner[i[0]]) != i[1]]
            edge_list.extend(edges)
            
            idx_0 += 1
        
        edge_set = set(edge_list)
        subgraph.add_edge_list(list(edge_set))

        print "Solve connected subgraphs..."
        ccs = subgraph.get_components(min_vertices=cc_min_vertices,
                                      output_folder="./ccs/",
                                      return_graphs=True)

        j = 0
        solutions = []
        for cc in ccs:
            cc.g.reindex_edges()
            cc_solution = solve(cc,
                                start_edge_prior,
                                distance_factor,
                                orientation_factor,
                                comb_angle_factor,
                                selection_cost,
                                time_limit,
                                output_dir=None,
                                voxel_size=None,
                                z_correction=0,
                                chunk_shift=np.array([0.,0.,0.]))

            j += 1

            solutions.append(cc_solution)
        
        return solutions

    def write_solution(self,
                       solution,
                       index_map,
                       name_db,
                       collection):
        """
        Add solved edges to collection and
        remove degree 0 vertices.
        Index map needs to specify local to
        global vertex index:
        {local: global}
        """
        
        graph = self._get_client(name_db, collection, overwrite=False)

        print "Write solution..."
        index_map[-1] = -1
        index_map_get = np.vectorize(index_map.get)
        edge_array = solution.get_edge_array()
        if edge_array.size:
            edges_global = index_map_get(np.delete(edge_array, 2, 1))

            print "Insert solved edges..."
            for edge_id in edges_global:
                edge = deepcopy(self.template_edge)
            
                edge["id_v0_global"] = str(edge_id[0][0]) # convert uint64 to str, faster & mdb doesn't accept 64 bit int
                edge["id_v1_global"] = str(edge_id[1][0])
                edge["id_v0_mongo"] = edge_id[0][1]
                edge["id_v1_mongo"] = edge_id[1][1]

                graph.insert_one(edge)
        else:
            print "No edges in solution, skip..."

        print "Remove degree 0 vertices..."
        vertex_degrees = np.array(solution.g.degree_property_map("total").a)
        vertices_deg_0_global = index_map_get(np.where(vertex_degrees == 0))
 
        graph.delete_many({"_id": {"$in": [v_id[1] for v_id in vertices_deg_0_global[0,:]]}})
        
        print "...Done"  
        
         
    def save_candidates(self,
                        name_db,
                        prob_map_stack_chunk,
                        offset_chunk,
                        gs,
                        ps,
                        voxel_size,
                        id_offset,
                        collection="graph",
                        overwrite=False):

        graph = self._get_client(name_db, collection, overwrite=overwrite)

        candidates = extract_candidates(prob_map_stack_chunk,
                                        gs,
                                        ps,
                                        voxel_size,
                                        bounding_box=None,
                                        bs_output_dir=None,
                                        offset_pos=offset_chunk)

        print "Write candidate vertices..."
                
        vertex_id = id_offset
        for candidate in candidates:
            pos_phys = np.array([candidate.position[j] * voxel_size[j] for j in range(3)])
            ori_phys = np.array([candidate.orientation[j] * voxel_size[j] for j in range(3)])
            partner = candidate.partner_identifier

            vertex = deepcopy(self.template_vertex)
            
            vertex["px"] = pos_phys[0]
            vertex["py"] = pos_phys[1]
            vertex["pz"] = pos_phys[2]
            vertex["ox"] = ori_phys[0]
            vertex["oy"] = ori_phys[1]
            vertex["oz"] = ori_phys[2]
            vertex["id_partner_global"] = partner
            vertex["id_global"] = vertex_id
        
            id_mongo = graph.insert_one(vertex).inserted_id
            vertex_id += 1


    def save_g1_graph(self, g1_graph, name_db, collection="graph", overwrite=False):
        """
        db/gt 
        """
        print "Update database..."

        client = MongoClient()

        db = client[name_db]
        collections = db.collection_names()
        
        if collection in collections:
            if overwrite:
                print "Warning, overwrite {}.{}!".format(name_db, collection)
                self.create_collection(name_db=name_db, 
                                       collection=collection, 
                                       overwrite=True)

            print "Collection already exists, insert in {}.{}...".format(name_db, collection)
        else:
            print "Database empty, create {}.{}...".format(name_db, collection)
            
        graph = db[collection]

        vertex_positions = g1_graph.get_position_array().T
        vertex_orientations = g1_graph.get_orientation_array().T
        partner = g1_graph.get_vertex_property("partner").a
        edges = g1_graph.get_edge_array()

        print "Insert vertices..."

        index_map = {}
        vertices = []
        vertex_id = 0
        for pos, ori, partner in zip(vertex_positions,
                                     vertex_orientations,
                                     partner):

            vertex = deepcopy(self.template_vertex)

            vertex["px"] = pos[0]
            vertex["py"] = pos[1]
            vertex["pz"] = pos[2]
            vertex["ox"] = ori[0]
            vertex["oy"] = ori[1]
            vertex["oz"] = ori[2]
            vertex["id_partner_global"] = partner
            vertex["id_global"] = vertex_id
        
            id_mongo = graph.insert_one(vertex).inserted_id
            index_map[vertex_id] = id_mongo 
            vertex_id += 1

        print "Insert edges..."

        for edge_id in edges:
            edge_mongo = [index_map[edge_id[0]], index_map[edge_id[1]]]
            edge = deepcopy(self.template_edge)
            
            edge["id_v0_global"] = str(edge_id[0]) # convert uint64 to str, faster & mdb doesn't accept 64 bit int
            edge["id_v1_global"] = str(edge_id[1])
            edge["id_v0_mongo"] = edge_mongo[0]
            edge["id_v1_mongo"] = edge_mongo[1]

            graph.insert_one(edge)
