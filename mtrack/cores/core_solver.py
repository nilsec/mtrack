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

from mtrack.solve import solve


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
            partner = [index_map[p] for p in partner]
            partner = np.array(partner)
            g1.set_partner(0,0, vals=partner)

        n = 0
        for e in edges:
            try:
                e0 = index_map[np.uint64(e["id_v0_global"])]
                e1 = index_map[np.uint64(e["id_v1_global"])]
                g1.add_edge(e0, e1)
            except KeyError:
                print "Edge Key Error..."
                pdb.set_trace()

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
                       core_id,
                       save_connected,
                       output_dir,
                       voxel_size,
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
       
        if save_connected:
            connected_output_dir = os.path.join(output_dir, "core_graphs/connected")
            
            if not os.path.exists(connected_output_dir):
                os.makedirs(connected_output_dir)

            g1_to_nml(subgraph, 
                      os.path.join(connected_output_dir, "core_connected_{}.nml".format(core_id)), 
                      knossos=True, 
                      voxel_size=voxel_size)

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
            #cc.g.purge_vertices()
            #cc.g.purge_edges()
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
        if graph.find({"core_id": core_id}).count() == 0:
            core = deepcopy(self.template_core)
            core["core_id"] = core_id
            graph.insert_one(core)

    def core_finished(self,
                      core_id,
                      name_db,
                      collection):
        
        graph = self._get_client(name_db, collection, overwrite=False)
        if graph.find({"core_id": core_id}).count() >= 1:
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
                degrees = []
                edge_pos = np.array([0.,0.,0.])
                inside = False
                for v in vedges:
                    edge_pos += np.array([v["px"], v["py"], v["pz"]])
                    degrees.append(v["degree"])

                if x_lim is not None:
                    edge_pos /= 2.
                    if np.all(edge_pos >= min_lim) and np.all(edge_pos <= max_lim):
                        inside = True
                else:
                    inside = True
                
                # Check if edge already exists (can happen if cores overlap)
                # We check that by checking the vertex degrees of the 
                # vertices of the edge. If one of them has degree == 2
                # the edge has to exist already because the database only holds
                # solved, correct edges thus all entities are paths.
                # Old:
                # if v_in == 2 and np.all(np.array(degrees)<=1):
                # this leaves the option for double edges of the kind o---o
                # New:
                if inside and np.any(np.array(degrees) == 0) and np.all(np.array(degrees)) <= 1:
                    graph.update_many({"_id": {"$in": [edge_id[0][1], edge_id[1][1]]}}, 
                                      {"$inc": {"degree": 1}, "$set": {"id_partner_global": -1}},
                                      upsert=False)

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
                                        offset_pos=offset_chunk,
                                        identifier_0=id_offset)

        print "Write candidate vertices..."
                
        #vertex_id = id_offset
        for candidate in candidates:
            pos_phys = np.array([candidate.position[j] * voxel_size[j] for j in range(3)])
            ori_phys = np.array([candidate.orientation[j] * voxel_size[j] for j in range(3)])
            partner = candidate.partner_identifier
            vertex_id = candidate.identifier
            print "vertex_id", vertex_id
            print "partner_id", partner
           
            if partner != (-1): 
                assert(abs(vertex_id - partner) <= 1)

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
            #vertex_id += 1

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