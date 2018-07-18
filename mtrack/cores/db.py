import numpy as np
import os
import json
from shutil import copyfile
import time
from copy import deepcopy
import glob
import h5py
from pymongo import MongoClient, IndexModel, ASCENDING
from scipy.spatial import KDTree
import itertools
from functools import partial

from mtrack.graphs import g1_graph
from mtrack.preprocessing import extract_candidates, connect_graph_locally

class DB(object):
    def __init__(self):
        self.vertex = {"px": None,
                       "py": None,
                       "pz": None,
                       "ox": None,
                       "oy": None,
                       "oz": None,
                       "id_partner": None,
                       "id": None,
                       "degree": None,
                       "solved": False,
                       "selected": False,
					   "type": "vertex",
                       "time_selected": [],
                       "by_selected": []}

        self.edge = {"id0": None,
                     "id1": None,
                     "solved": False,
                     "selected": False,
					 "type": "edge",
                     "time_selected": [],
                     "by_selected": []}


    def get_db(self, name_db):
        client = MongoClient()
        db = client[name_db]
        return db

    def get_client(self, name_db, collection, overwrite=False):
        print "Get client..."
        client = MongoClient(connect=False)

        db = self.get_db(name_db)
        collections = db.collection_names()
        
        if collection in collections:
            if overwrite:
                print "Warning, overwrite collection!"
                self.create_collection(name_db=name_db, 
                                       collection=collection, 
                                       overwrite=True)

                # Check that collection is empty after overwrite:
                assert(db[collection].find({}).count() == 0)

            print "Collection already exists, request {}.{}...".format(name_db, collection)
        else:
            print "Database empty, create...".format(name_db, collection)
            self.create_collection(name_db=name_db, 
                                   collection=collection, 
                                   overwrite=False)
            
        graph = db[collection]
        
        return graph

    def create_collection(self, name_db, collection, overwrite=False):
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

        graph.create_index([("id0", ASCENDING), ("id1", ASCENDING)], 
                           name="vedge_id", 
                           sparse=True)

    def get_g1(self,
               name_db,
               collection,
               x_lim,
               y_lim,
               z_lim,
               query_edges=True):

        vertices, edges = self.__get_roi(name_db,
                                         collection,
                                         x_lim,
                                         y_lim,
                                         z_lim,
                                         query_edges)

        g1, index_map = self.__roi_to_g1(vertices, edges)


        return g1, index_map


    def get_selected(self,
					 name_db,
					 collection,
					 x_lim,
					 y_lim,
					 z_lim):

        vertices, edges = self.__get_roi(name_db,
                                         collection,
                                         x_lim,
                                         y_lim,
                                         z_lim,
                                         query_edges=True)

        selected_vertices = []
        for v in vertices:
            if v["selected"]:
                v["id_partner"] = -1
                selected_vertices.append(v)

        selected_edges = []
        for e in edges:
            if e["selected"]:
                selected_edges.append(e)

        g1_selected, index_map = self.__roi_to_g1(selected_vertices, 
                                                  selected_edges)

        return g1_selected, index_map


    def validate_selection(self, 
                           name_db,
                           collection,
                           x_lim,
                           y_lim,
                           z_lim):

        print "Validate Solution..."
        g1_selected, index_map = self.get_selected(name_db,
                                                   collection,
                                                   x_lim,
                                                   y_lim,
                                                   z_lim)

        for v in g1_selected.get_vertex_iterator():
            assert(len(g1_selected.get_incident_edges(v)) <= 2),\
                   "Selection has branchings"

        print "...No violations"

        
    def write_solution(self,
                       solution,
                       index_map,
                       name_db,
                       collection,
                       x_lim=None,
                       y_lim=None,
                       z_lim=None,
                       id_writer=-1):
        
        graph = self.get_client(name_db, collection, overwrite=False)

        print "Write solution..."
        if x_lim is not None and y_lim is not None and z_lim is not None:
            min_lim = np.array([x_lim["min"], y_lim["min"], z_lim["min"]])
            max_lim = np.array([x_lim["max"], y_lim["max"], z_lim["max"]])
        else:
            print "WARNING: No write ROI provided, write full graph"
            min_lim = np.array([])
            max_lim = np.array([])

        print "Write selected..."
        for e in solution.get_edge_iterator():
            v0_pos = np.array(solution.get_position(e.source()))
            v1_pos = np.array(solution.get_position(e.target()))
            e_pos = (v0_pos + v1_pos)/2.
 
            if np.all(e_pos >= min_lim) and np.all(e_pos < max_lim):
                v0_mapped = index_map[e.source()]
                v1_mapped = index_map[e.target()]
            
                graph.update_one({"$and": [{"id0": {"$in": [v0_mapped, v1_mapped]}},
                                           {"id1": {"$in": [v0_mapped, v1_mapped]}}]},
                                 {"$set": {"selected": True},
                                  "$push": {"by_selected": id_writer, 
                                            "time_selected": str(time.localtime(time.time()))}},
                                 upsert=False)

                # Edge implies vertices:
                graph.update_one({"id": v0_mapped},
                                 {"$inc": {"degree": 1},
                                  "$set": {"selected": True},
                                  "$push": {"by_selected": id_writer, 
                                            "time_selected": str(time.localtime(time.time()))}},
                                 upsert=False)


                graph.update_one({"id": v1_mapped},
                                 {"$inc": {"degree": 1},
                                  "$set": {"selected": True},
                                  "$push": {"by_selected": id_writer, 
                                            "time_selected": str(time.localtime(time.time()))}},
                                 upsert=False)
 
                assert(graph.find({"degree": {"$gte": 3}}).count()==0)

        print "Selected edges: ", graph.find({"selected": True, "type": "edge"}).count()
        print "Selected vertices: ", graph.find({"selected": True, "type": "vertex"}).count()

        # Set all edges and vertices in solve region to solved
        vertices, edges = self.__get_roi(name_db,
                                         collection,
                                         x_lim,
                                         y_lim,
                                         z_lim,
                                         query_edges=True)

        print "Write solved..."
        for v in vertices:
            graph.update_one({"id": v["id"]},
                             {"$set": {"solved": True}},
                             upsert=False)

        for e in edges:
            graph.update_one({"$and": [{"id0": e["id0"]}, {"id1": e["id1"]}]},
                             {"$set": {"solved": True}},
                             upsert=False)


    def write_candidates(self,
                         name_db,
                         prob_map_stack_chunk,
                         offset_chunk,
                         gs,
                         ps,
                         voxel_size,
                         id_offset,
                         collection,
                         overwrite=False):

        graph = self.get_client(name_db, collection, overwrite=overwrite)

        candidates = extract_candidates(prob_map_stack_chunk,
                                        gs,
                                        ps,
                                        voxel_size,
                                        bounding_box=None,
                                        bs_output_dir=None,
                                        offset_pos=offset_chunk,
                                        identifier_0=id_offset)

        print "Write candidate vertices..."
        for candidate in candidates:
            pos_phys = np.array([round(candidate.position[j]) * voxel_size[j] for j in range(3)])
            assert(np.all(np.array(pos_phys, dtype=int) == pos_phys))
            ori_phys = np.array([candidate.orientation[j] * voxel_size[j] for j in range(3)])

            partner = candidate.partner_identifier
            vertex_id = candidate.identifier
           
            if partner != (-1): 
                assert(abs(vertex_id - partner) <= 1)

            vertex = deepcopy(self.vertex)
            
            vertex["px"] = pos_phys[0]
            vertex["py"] = pos_phys[1]
            vertex["pz"] = pos_phys[2]
            vertex["ox"] = ori_phys[0]
            vertex["oy"] = ori_phys[1]
            vertex["oz"] = ori_phys[2]
            vertex["id_partner"] = partner
            vertex["id"] = vertex_id
            vertex["degree"] = 0
            vertex["selected"] = False
            vertex["solved"] = False

            graph.insert_one(vertex)

        return vertex_id

    def reset_collection(self,
                         name_db,
                         collection):

        print "Reset {}.{}...".format(name_db, collection)
        graph = self.get_client(name_db, collection)

        graph.update_many({}, 
                          {"$set": {"selected": False, 
                                    "solved": False, 
                                    "degree": 0,
                                    "time_selected": [],
                                    "by_selected": []}},
                          upsert=False)

        n_selected = graph.find({"selected": True}).count()
        n_solved = graph.find({"solved": True}).count()
        assert(n_selected == 0)
        assert(n_solved == 0)
    
    def connect_candidates(self,
                           name_db,
                           collection,
                           x_lim,
                           y_lim,
                           z_lim,
                           distance_threshold):

        g1_candidates, index_map = self.get_g1(name_db,
                                               collection,
                                               x_lim,
                                               y_lim,
                                               z_lim,
                                               query_edges=True)

        # Already present edges are not added again here:
        g1_connected = connect_graph_locally(g1_candidates,
                                             distance_threshold)

        graph = self.get_client(name_db,
                                collection)

        for e in g1_connected.get_edge_iterator():
            v0 = index_map[e.source()]
            v1 = index_map[e.target()]
            
            edge = deepcopy(self.edge)
            edge["id0"] = v0
            edge["id1"] = v1
            edge["selected"] = False
            edge["solved"] = False

            # Check if edge is already in db:
            e_N = graph.find({"$and": [{"id0": {"$in": [v0, v1]}},
                                       {"id1": {"$in": [v0, v1]}}]}).count()
            assert(e_N<=1)

            if e_N == 0:
                graph.insert_one(edge)


    def __get_roi(self,
                  name_db,
                  collection,
                  x_lim,
                  y_lim,
                  z_lim,
                  query_edges=True):

        print "Extract subgraph..."
        
        graph = self.get_client(name_db, collection, overwrite=False)

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

        vertex_ids = [v["id"] for v in vertices]
        
        
        edges = [] 
        if query_edges:
            print "Perform edge query..."
            edges = list(graph.find({"$and": [{"id0": {"$in": vertex_ids}},
                                              {"id1": {"$in": vertex_ids}}]}))
        
        if not vertices:
            print "Warning, requested region holds no vertices!"
        if not edges:
            print "Warning, requested region holds no edges!"

        return vertices, edges


    def __roi_to_g1(self, vertices, edges):
        if not vertices:
            raise ValueError("Vertex list is empty")

        g1 = g1_graph.G1(len(vertices), init_empty=False)
        # Init index map, global <-> local:
        index_map = {}
        index_map_inv = {}

        # Flag solved and selected edges & vertices:
        g1.new_vertex_property("selected", dtype="bool", value=False)
        g1.new_vertex_property("solved", dtype="bool", value=False)

        g1.new_edge_property("selected", dtype="bool", value=False)
        g1.new_edge_property("solved", dtype="bool", value=False)

        partner = {}
        n = 0
        for v in vertices:
            g1.set_position(n, np.array([v["px"], v["py"], v["pz"]]))
            g1.set_orientation(n, np.array([v["ox"], v["oy"], v["oz"]]))

            if v["selected"]:
                g1.select_vertex(n)

            if v["solved"]:
                g1.solve_vertex(n)

            partner[n] = v["id_partner"]
            index_map[v["id"]] = n
            index_map_inv[n] = v["id"]

            n += 1

        index_map[-1] = -1
        index_map_inv[-1] = -1

        for v in g1.get_vertex_iterator():
            g1.set_partner(v, index_map[partner[v]])

        for e in edges:
            v0 = index_map[np.uint64(e["id0"])]
            v1 = index_map[np.uint64(e["id1"])]
            e_g1 = g1.add_edge(v0, v1)
        
            if e["selected"]:
                g1.select_edge(e_g1)

            if e["solved"]:
                g1.solve_edge(e_g1)
        
        return g1, index_map_inv
