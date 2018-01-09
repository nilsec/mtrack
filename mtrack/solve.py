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
import itertools
from functools import partial
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


class CoreSolver(object):
    def __init__(self):
        self.template_vertex = {"px": None,
                                "py": None,
                                "pz": None,
                                "ox": None,
                                "oy": None,
                                "oz": None,
                                "id_partner_global": None,
                                "id_global": None,
                                "degree": None}


        self.template_edge = {"id_v0_global": None,
                              "id_v1_global": None,
                              "id_v0_mongo": None,
                              "id_v1_mongo": None}

        self.template_core = {"core_id": None}


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

    def _get_db(self, name_db):
        client = MongoClient()
        db = client[name_db]
        return db


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

        graph.create_index([("core_id", ASCENDING)],
                           name="core_id",
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
                     z_lim,
                     query_edges=True):

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
        
        
        edges = [] 
        if query_edges:
            print "Perform edge query..."
            edges = list(graph.find({"$and": [{"id_v0_mongo": {"$in": vertex_ids}},
                                              {"id_v1_mongo": {"$in": vertex_ids}}]}))
        
        print "...Done"

        if not vertices:
            print "Warning, requested region holds no vertices!"
        if not edges:
            print "Warning, requested region holds no edges!"

        return vertices, edges

    def _map_partner(self, index_map, x):
        try:
            return index_map[x]
        except KeyError:
            return -1
        

    def subgraph_to_g1(self, vertices, edges, set_partner=True):
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
        
        if set_partner:
            partner = [self._map_partner(index_map=index_map, x=p) for p in partner]
            partner = np.array(partner)
            g1.set_partner(0,0, vals=partner)

        n = 0
        for e in edges:
            try:
                e0 = index_map[np.uint64(e["id_v0_global"])]
                e1 = index_map[np.uint64(e["id_v1_global"])]
                g1.add_edge(e0, e1)
            except KeyError:
                pass

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
                       write=True,
                       hcs=False):

        print "Solve subgraph..."

        print "Connect locally..."
        positions = subgraph.get_position_array().T
        partner = subgraph.get_partner_array()

        print "Get vertex degrees..."
        vertex_degrees = np.array(subgraph.g.degree_property_map("total").a)
        vertex_mask_0 = vertex_degrees == 0
        vertex_mask_1 = vertex_degrees <= 1

        print "Build index map..."
        index_map_0 = np.cumsum(vertex_mask_0) - 1
        index_map_1 = np.cumsum(vertex_mask_1) - 1

        enum_0 = np.arange(len(index_map_0))
        enum_1 = np.arange(len(index_map_1))

        # Mask both arrays:
        index_map_0 = index_map_0[vertex_mask_0]
        index_map_1 = index_map_1[vertex_mask_1]

        enum_0 = enum_0[vertex_mask_0]
        enum_1 = enum_1[vertex_mask_1]

        # Create dict
        index_map_0 = dict(zip(index_map_0, enum_0))
        index_map_1 = dict(zip(index_map_1, enum_1))

        """
        index_map_0 = {sum(vertex_mask_0[:i]):i for i in range(len(vertex_mask_0)) if vertex_mask_0[i]}
        index_map_1 = {sum(vertex_mask_1[:j]):j for j in range(len(vertex_mask_1)) if vertex_mask_1[j]}
        """
        
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

        """
        kdtree = KDTree(positions_0)
        pairs = kdtree.query_pairs(r=distance_threshold, p=2.0, eps=0)

        for edge in pairs:
            if subgraph.get_partner(index_map_0[edge[0]]) != index_map_0[edge[1]]:
                subgraph.add_edge(index_map_0[edge[0]], index_map_0[edge[1]])
        """
        
        g1_to_nml(subgraph, "./sugbgraph_connected.nml", knossos=True, voxel_size=[5.,5.,50.])

        print "Solve connected subgraphs..."
        if hcs:
            subgraph.new_edge_property("weight", "int", vals=np.ones(subgraph.get_number_of_edges()))
            ccs = subgraph.get_hcs(subgraph, remove_singletons=4)
        else:
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

    def finish_core(self,
                    core_id,
                    name_db,
                    collection):

        graph = self._get_client(name_db, collection, overwrite=False)
        
        core = deepcopy(self.template_core)
        core["core_id"] = core_id
        
        graph.insert_one(core)

    def core_finished(self,
                      core_id,
                      name_db,
                      collection):
        
        graph = self._get_client(name_db, collection, overwrite=False)
        if graph.find({"core_id": core_id}).count() == 1:
            return True
        else:
            return False
        
    def write_solution(self,
                       solution,
                       index_map,
                       name_db,
                       collection,
                       x_lim=None,
                       y_lim=None,
                       z_lim=None,
                       n=0):
        """
        Add solved edges to collection and
        remove degree 0 vertices.
        Index map needs to specify local to
        global vertex index:
        {local: global}
        """
        graph = self._get_client(name_db, collection, overwrite=False)

        print "Write solution..."
        if x_lim is not None:
            min_lim = np.array([x_lim["min"], y_lim["min"], z_lim["min"]])
            max_lim = np.array([x_lim["max"], y_lim["max"], z_lim["max"]])
        else:
            min_lim = np.array([])
            max_lim = np.array([]) 

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

                
            
                # Check if edge lies in limits
                vedges = graph.find({"_id": {"$in": [edge_id[0][1], edge_id[1][1]]}})
                v_in = 0
                degrees = []
                for v in vedges:
                    pos = np.array([v["px"], v["py"], v["pz"]])
                    degrees.append(v["degree"])
                    if x_lim is not None:
                        if np.all(pos > min_lim) and np.all(pos < max_lim):
                            v_in += 1
                    else:
                        v_in += 1
                
                # Check if edge already exists (can happen if cores overlap)
                # We check that by checking the vertex degrees of the 
                # vertices of the edge. If one of them has degree == 2
                # the edge has to exist already because the database only holds
                # solved, correct edges thus all entities are paths.
                # Old:
                # if v_in == 2 and np.all(np.array(degrees)<=1):
                # this leaves the option for double edges of the kind o---o
                # New:
                if v_in == 2 and np.any(np.array(degrees) == 0):
                    graph.update_many({"_id": {"$in": [edge_id[0][1], edge_id[1][1]]}}, 
                                      {"$inc": {"degree": 1}}, upsert=False)

                    graph.insert_one(edge)
        else:
            print "No edges in solution, skip..."
        

    def remove_deg_0_vertices(self, 
                              name_db,
                              collection,
                              x_lim,
                              y_lim,
                              z_lim):

        print "Remove degree 0 vertices..."

        graph = self._get_client(name_db,
                                 collection)
        graph.delete_many({"$and": 
                                    [   
                                        {   
                                         "pz": {"$gte": z_lim["min"],
                                                 "$lt": z_lim["max"]},
                                         "py": {"$gte": y_lim["min"],
                                                 "$lt": y_lim["max"]},
                                         "px": {"$gte": x_lim["min"],
                                                 "$lt": x_lim["max"]}
                                    
                                        },
                                        {
                                         "degree": 0
                                        }
                                    ]
                            })

         
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
            vertex["degree"] = 0
        
            id_mongo = graph.insert_one(vertex).inserted_id
            vertex_id += 1

        return vertex_id


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


class CoreScheduler(object):
    def __init__(self, cores):
        self.cores = cores

        self.running = set()
        self.finished = set()

    def request_core(self):
        for core in self.cores:
            if not core.id in self.finished:
                if not core.id in self.running:
                    core_nbs = core.nbs
                    if not (self.running & core_nbs):
                        self.running.add(core.id)
                
                        return core

        return None

    def finish_core(self, core_id):
        self.finished.add(core_id)
        self.running.remove(core_id)


class CoreBuilder(object):
    def __init__(self, 
                 volume_size,
                 core_size,
                 context_size,
                 min_core_overlap=np.array([0,0,0]),
                 offset=np.array([0,0,0])):

        self.volume_size = np.array(volume_size, dtype=float)
        self.core_size = np.array(core_size, dtype=float)
        self.context_size = np.array(context_size, dtype=float)
        self.min_core_overlap = np.array(min_core_overlap, dtype=float)

        self.offset = offset

        self.block_size = self.core_size + self.context_size

        self.cores = []
        self.running = []

    
    def _gen_nbs(self, core_id, n_cores):
        """
        Generate all 25 nbs of a core.
        Seemed easy, edge cases make this function
        a beast. In hindsight the edge case exclusions
        probably define the neighbours sufficiently
        and f_core is obsolete. 
        I.e. neighbours are those ids 
        that differ in zy plane for and are not more than
        mod 1 distant in the +- cases. 
        For the 0 case same with mod but stay in same plane.
        """
        n_Z = n_cores[2]
        n_Y = n_cores[1]
        n_X = n_cores[0]

        # Corresponds to x=0 plane
        f_core_0 = lambda j,k: core_id + j + n_Z * k
        
        # x - 1 plane
        f_core_m1 = lambda j,k: core_id + j - ((n_Y + k) * n_Z)

        # x + 1 plane
        f_core_p1 = lambda j,k: core_id + j + ((n_Y + k) * n_Z)

        core_id_range = range(reduce(lambda x,y: x * y, n_cores))

        nbs = set()
        for j in [0,1,-1]:
            for k in [0,1,-1]:
                c0 = f_core_0(j,k)
                cm1 = f_core_m1(j,k)
                cp1 = f_core_p1(j,k)

                if c0 in core_id_range:
                    # Generally the id needs to be in range (1,1,1)
                    if np.all(np.array([abs(c0%n_Z - core_id%n_Z), 
                                        abs(c0%n_Y - core_id%n_Y)]) <= np.array([1,1])):
                
                        if abs(c0/(n_Y * n_Z) - core_id/(n_Y * n_Z)) == 0:
  
                            # In case of +- 1 only z plane change
                            if abs(c0 - core_id) > 1:
                                # id needs to change y+-1 plane
                                if abs(c0/n_Z - core_id/n_Z) == 1:
                                    nbs.add(c0)
                            else:
                                nbs.add(c0)
               
                if np.all(np.array([abs(cm1%n_Z - core_id%n_Z), 
                                    abs(cm1%n_Y - core_id%n_Y)]) <= np.array([1,1])):
                    if cm1 in core_id_range:
                        # id needs to change x-1 plane
                        if abs(cm1/(n_Y * n_Z) - core_id/(n_Y * n_Z)) == 1:
                            nbs.add(cm1)


                if np.all(np.array([abs(cp1%n_Z - core_id%n_Z), 
                                    abs(cp1%n_Y - core_id%n_Y)]) <= np.array([1,1])):
                    if cp1 in core_id_range:
                        # id needs to change to x+1 plane
                        if abs(cp1/(n_Y * n_Z) - core_id/(n_Y * n_Z)) == 1:
                            nbs.add(cp1)
 
        nbs.remove(core_id)

        return nbs

    def generate_cores(self):
        print "Generate cores..."
        if np.all(self.volume_size == self.core_size):
            cores = [Core(x_lim={"min": self.offset[0], 
                               "max": self.core_size[0] + self.offset[0]},
                        y_lim={"min": self.offset[1], 
                               "max": self.core_size[1] + self.offset[1]},
                        z_lim={"min": self.offset[2], 
                               "max": self.core_size[2] + self.offset[2]},
                        context=[0.0,0.0,0.0],
                        core_id=0,
                        nbs=set([]))]
            return cores
            

        n_cores, ovlp = self._get_overlap()
        max_ids = reduce(lambda x,y: x*y, n_cores)        

        x_0 = self.context_size[0]
        y_0 = self.context_size[1]
        z_0 = self.context_size[2]


        cores = []
        core_id = 0
        for x_core in range(n_cores[0]):
            x_0 = self.context_size[0] + x_core * self.core_size[0] - x_core * ovlp[0]
 
            for y_core in range(n_cores[1]):
                y_0 = self.context_size[1] + y_core * self.core_size[1] - y_core * ovlp[1]
 
                for z_core in range(n_cores[2]):
                    z_0 = self.context_size[2] + z_core * self.core_size[2] - z_core * ovlp[2]

                    nbs = self._gen_nbs(core_id, n_cores)
                                            

                    core = Core(x_lim={"min": x_0 + self.offset[0], 
                                       "max": x_0 + self.core_size[0] + self.offset[0]},
                                y_lim={"min": y_0 + self.offset[1], 
                                       "max": y_0 + self.core_size[1] + self.offset[1]},
                                z_lim={"min": z_0 + self.offset[2], 
                                       "max": z_0 + self.core_size[2] + self.offset[2]},
                                context=self.context_size,
                                core_id=core_id,
                                nbs=nbs)

                    cores.append(core)
                    core_id += 1

        return cores


    def _get_overlap(self):
        """
        Get actual overlap such that the whole volume is packed with
        cubes of size core_size + context size with a minumum
        core overlap of self.min_core_overlap. The calculated
        overlap will be >= self.min_core_overlap.
        """
        core_volume = self.volume_size - 2*self.context_size
        n_cores = np.ceil(core_volume/self.core_size)
        assert(np.all(n_cores > 1))
        core_volume_novlp = n_cores * self.core_size
        diff = core_volume_novlp - core_volume
        ovlp = diff/(n_cores - 1.0)
 
        while np.any(ovlp < self.min_core_overlap):
            core_volume_novlp = n_cores * self.core_size
            diff = core_volume_novlp - core_volume
            ovlp = diff/(n_cores - 1)
            n_cores[np.where(ovlp < self.min_core_overlap)] += 1
        
        for i in range(3):
            for j in range(3):
                balance = ovlp[i]/ovlp[j]
                if balance < 0.7:
                    print "WARNING, overlap not balanced!"

        return n_cores.astype(int), ovlp
        

class Core(object):
    def __init__(self,
                 x_lim,
                 y_lim,
                 z_lim,
                 context,
                 core_id,
                 nbs):
        
        self.x_lim_core = x_lim
        self.y_lim_core = y_lim
        self.z_lim_core = z_lim

        self.x_lim_context = {"min": x_lim["min"] - context[0], 
                              "max": x_lim["max"] + context[0]}

        self.y_lim_context = {"min": y_lim["min"] - context[1], 
                              "max": y_lim["max"] + context[1]}
 
        self.z_lim_context = {"min": z_lim["min"] - context[2], 
                              "max": z_lim["max"] + context[2]}
        
        self.id = core_id
        self.nbs = nbs
