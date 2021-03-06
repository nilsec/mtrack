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
import logging
from mtrack.evaluation import DDA3

from mtrack.graphs import g1_graph
from mtrack.preprocessing import connect_graph_locally
import ConfigParser

logger = logging.getLogger(__name__)


class DB(object):
    def __init__(self, credentials):
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
                     "cost": None,
                     "solved": False,
                     "selected": False,
		     "type": "edge",
                     "time_selected": [],
                     "by_selected": []}

        with open(credentials) as fp:
            config = ConfigParser.ConfigParser()
            config.readfp(fp)
            self.credentials = {}
            self.credentials["user"] = config.get("Credentials", "user")
            self.credentials["password"] = config.get("Credentials", "password")
            self.credentials["host"] = config.get("Credentials", "host")
            self.credentials["port"] = config.get("Credentials", "port")

        self.auth_string = 'mongodb://{}:{}@{}:{}'.format(self.credentials["user"],
                                                          self.credentials["password"],
                                                          self.credentials["host"],
                                                          self.credentials["port"])

    def get_client(self):
        client = MongoClient(self.auth_string, connect=False)
        return client

    def get_db(self, name_db):
        client = self.get_client()
        db = client[name_db]
        return db

    def get_collection(self, name_db, collection, overwrite=False):
        logger.info("Get client...")
        client = self.get_client()

        db = self.get_db(name_db)
        collections = db.collection_names()
        
        if collection in collections:
            if overwrite:
                logger.info("Warning, overwrite collection {}...".format(collection))
                self.create_collection(name_db=name_db, 
                                       collection=collection, 
                                       overwrite=True)

                # Check that collection is empty after overwrite:
                assert(db[collection].find({}).count() == 0)

            else:
                logger.info("Collection already exists, request {}.{}...".format(name_db, collection))
        else:
            logger.info("Collection does not exist, create...")
            self.create_collection(name_db=name_db, 
                                   collection=collection, 
                                   overwrite=False)
            
        graph = db[collection]
        
        return graph

    def create_collection(self, name_db, collection, overwrite=False):
        logger.info("Create new db collection {}.{}...".format(name_db, collection))
        client = self.get_client()
        db = self.get_db(name_db)
        
        if overwrite:
            logger.info("Overwrite {}.{}...".format(name_db, collection))
            db.drop_collection(collection)

        graph = db[collection]

        logger.info("Generate indices...")
        graph.create_index([("pz", ASCENDING), ("py", ASCENDING), ("px", ASCENDING)], 
                           name="pos", 
                           sparse=True)

        graph.create_index([("id0", ASCENDING), ("id1", ASCENDING)], 
                           name="vedge_id", 
                           sparse=True)

    def add_edge_cost(self,
                      name_db,
                      collection,
                      voxel_size,
                      volume_offset,
                      prob_map_file,
                      prob_map_dset):

        logger.info("Add edge cost...")

        graph = self.get_collection(name_db, collection, overwrite=False)

        voxel_size = np.array(voxel_size)
        f = h5py.File(prob_map_file, "r")
        prob_map = np.array(f[prob_map_dset])
        attrs = f[prob_map_dset].attrs.items()
        f.close()

        shape = np.shape(prob_map)
        chunk_limits = attrs[1][1]
        offset_chunk = [chunk_limits[0][0],
                        chunk_limits[1][0],
                        chunk_limits[2][0]]

        #offset_chunk += np.array(volume_offset)
        offset_chunk *= np.array(voxel_size, dtype=int)

        size_chunk = np.array([shape[2], shape[1], shape[0]]) * voxel_size
        x_lim = {"min": offset_chunk[0], "max": offset_chunk[0] + size_chunk[0]}
        y_lim = {"min": offset_chunk[1], "max": offset_chunk[1] + size_chunk[1]}
        z_lim = {"min": offset_chunk[2], "max": offset_chunk[2] + size_chunk[2]}

        g1, index_map = self.get_g1(name_db,
                                    collection,
                                    x_lim,
                                    y_lim,
                                    z_lim)

        if g1 is not None:
            n_edges = g1.get_number_of_edges()
            i = 1
            edge_to_cost = {}

            logger.info("Overlay edges with prob map...")
            for e in g1.get_edge_iterator():
                start = np.array(g1.get_position(e.source())/voxel_size, dtype=int)
                end = np.array(g1.get_position(e.target())/voxel_size, dtype=int)
                dda3 = DDA3(start, end, scaling=voxel_size)
                line = dda3.draw()

                cost = 0.0
                for p in line:
                    p -= (offset_chunk/voxel_size).astype(int)
                    cost += (1. - prob_map[p[2], p[1], p[0]])
                cost /= len(line)

                v0_mapped = index_map[int(e.source())]
                v1_mapped = index_map[int(e.target())]

                graph.update_one({"$and": [{"id0": {"$in": [v0_mapped, v1_mapped]}},
                                           {"id1": {"$in": [v0_mapped, v1_mapped]}}]},
                                 {"$set": {"cost": cost}},
                                 upsert=False)

                if i % 100 == 0:
                    logger.info(str(float(i)/n_edges * 100) +  "% done")
                    logger.info("cost_write: " + str(cost))
                i += 1

    def get_g1(self,
               name_db,
               collection,
               x_lim,
               y_lim,
               z_lim,
               query_edges=True):

        vertices, edges = self.__get_vertex_roi(name_db,
                                                collection,
                                                x_lim,
                                                y_lim,
                                                z_lim,
                                                query_edges)

        if vertices:
            g1, index_map = self.__roi_to_g1(vertices, edges)
        else:
            g1 = None
            index_map = None


        return g1, index_map


    def get_selected(self,
                     name_db,
                     collection,
                     x_lim,
                     y_lim,
                     z_lim):

        vertices, edges = self.__get_vertex_roi(name_db,
                                                collection,
                                                x_lim,
                                                y_lim,
                                                z_lim,
                                                query_edges=True)

        selected_vertices = set()
        id_to_vertex = {}
        for v in vertices:
            id_to_vertex[v["id"]] = v
            if v["selected"]:
                v["id_partner"] = -1
                selected_vertices.add(v["id"])

        selected_edges = []
        for e in edges:
            if e["selected"]:
                selected_edges.append(e)
                """
                This is necessary to avoid the retrieval of selected edges
                for which only one of the vertices is selected yet.
                This would lead to errors downstream.
                """
                selected_vertices.add(e["id0"])
                selected_vertices.add(e["id1"])
                id_to_vertex[e["id0"]]["id_partner"] = -1
                id_to_vertex[e["id1"]]["id_partner"] = -1
        

        g1_selected, index_map = self.__roi_to_g1([id_to_vertex[v_id] for v_id in selected_vertices], 
                                                  selected_edges)

        return g1_selected, index_map


    def validate_selection(self, 
                           name_db,
                           collection,
                           x_lim,
                           y_lim,
                           z_lim):

        logger.info("Validate Selection...")
        g1_selected, index_map = self.get_selected(name_db,
                                                   collection,
                                                   x_lim,
                                                   y_lim,
                                                   z_lim)

        for v in g1_selected.get_vertex_iterator():
            assert(len(g1_selected.get_incident_edges(v)) <= 2),\
                   "Selection has branchings"

        logger.info("...No violations")
        return g1_selected

        
    def write_solution(self,
                       solution,
                       index_map,
                       name_db,
                       collection,
                       x_lim=None,
                       y_lim=None,
                       z_lim=None,
                       id_writer=-1):
        
        graph = self.get_collection(name_db, collection, overwrite=False)

        logger.info("Write solution...")
        if x_lim is not None and y_lim is not None and z_lim is not None:
            min_lim = np.array([x_lim["min"], y_lim["min"], z_lim["min"]])
            max_lim = np.array([x_lim["max"], y_lim["max"], z_lim["max"]])
        else:
            logger.warning("WARNING: No write ROI provided, write full graph")
            min_lim = np.array([])
            max_lim = np.array([])

        logger.info("Write selected...")
        for e in solution.get_edge_iterator():
            v0_id_db = index_map[e.source()]
            v1_id_db = index_map[e.target()]

            """
            The position of the vertex with the lower global id
            decides about wether to write the edge or not.
            """
            assert(v0_id_db != v1_id_db)
            if v0_id_db < v1_id_db:
                deciding_vertex = e.source()
            else:
                deciding_vertex = e.target()
 
            v0_pos = np.array(solution.get_position(e.source()))
            v1_pos = np.array(solution.get_position(e.target()))
            deciding_vertex_pos = np.array(solution.get_position(deciding_vertex))
            assert(np.all(deciding_vertex_pos == np.array(deciding_vertex_pos, dtype=int)))
 
            if np.all(deciding_vertex_pos >= min_lim) and np.all(deciding_vertex_pos < max_lim):
                v0_mapped = index_map[e.source()]
                v1_mapped = index_map[e.target()]
            
                graph.update_one({"$and": [{"id0": {"$in": [v0_mapped, v1_mapped]}},
                                           {"id1": {"$in": [v0_mapped, v1_mapped]}}]},
                                 {"$set": {"selected": True, "solved": True},
                                  "$push": {"by_selected": id_writer, 
                                            "time_selected": str(time.localtime(time.time()))}},
                                 upsert=False)
                """
                Edge implies vertex if the vertex is in the same core.
                Otherwise we get conflicts in write/read access.
                """
                if np.all(v0_pos >= min_lim) and np.all(v0_pos < max_lim):
                    graph.update_one({"id": v0_mapped},
                                     {"$inc": {"degree": 1},
                                      "$set": {"selected": True, "solved": True},
                                      "$push": {"by_selected": id_writer, 
                                                "time_selected": str(time.localtime(time.time()))}},
                                     upsert=False)

                if np.all(v1_pos >= min_lim) and np.all(v1_pos < max_lim):
                    graph.update_one({"id": v1_mapped},
                                     {"$inc": {"degree": 1},
                                      "$set": {"selected": True, "solved": True},
                                      "$push": {"by_selected": id_writer, 
                                                "time_selected": str(time.localtime(time.time()))}},
                                     upsert=False)
                
                assert(graph.find({"degree": {"$gte": 3}}).count()==0)

        logger.info("Selected edges: {}".format(graph.find({"selected": True, "type": "edge"}).count()))
        logger.info("Selected vertices: {}".format(graph.find({"selected": True, "type": "vertex"}).count()))


    def write_solved(self,
                     name_db,
                     collection,
                     core):
        
        graph = self.get_collection(name_db, collection, overwrite=False)


        """
        The vertex roi does not capture the edges 
        that lie on the boarders of the core.
        Thus we grab the whole context and check 
        for bounds.
        """
        vertices, edges = self.__get_vertex_roi(name_db,
                                                collection,
                                                core.x_lim_context,
                                                core.y_lim_context,
                                                core.z_lim_context,
                                                query_edges=True)

        core_min = np.array([core.x_lim_core["min"], 
                             core.y_lim_core["min"],
                             core.z_lim_core["min"]])

        core_max = np.array([core.x_lim_core["max"],
                             core.y_lim_core["max"],
                             core.z_lim_core["max"]])

        logger.info("Write solved...")
        id_to_vertex_pos = {}
        for v in vertices:
            v_pos = np.array([v["px"], v["py"], v["pz"]])
            if np.all(v_pos>=core_min) and np.all(v_pos<core_max):
                graph.update_one({"id": v["id"]},
                                 {"$set": {"solved": True}},
                                 upsert=False)

            id_to_vertex_pos[v["id"]] = v_pos

        for e in edges:
            v0_id = e["id0"]
            v1_id = e["id1"]
           
            assert(v0_id != v1_id) 
            if v0_id < v1_id:
                deciding_vertex_id = v0_id
            else:
                deciding_vertex_id = v1_id

            deciding_vertex_pos = id_to_vertex_pos[deciding_vertex_id]
            if np.all(deciding_vertex_pos >= core_min) and\
               np.all(deciding_vertex_pos < core_max):

                graph.update_one({"$and": [{"id0": e["id0"]}, {"id1": e["id1"]}]},
                                 {"$set": {"solved": True}},
                                 upsert=False)
     

    def write_candidates(self,
                         name_db,
                         collection,
                         candidates,
                         voxel_size,
                         overwrite=False):


        graph = self.get_collection(name_db, collection, overwrite=overwrite)

        logger.info("Write candidate vertices...")
        vertex_id = graph.find({"selected": {"$exists": True}}).count()
        vertices = []
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
            vertices.append(vertex)

        if vertices:
            graph.insert_many(vertices)

        return vertex_id

    def reset_collection(self,
                         name_db,
                         collection):

        logger.info("Reset {}.{}...".format(name_db, collection))
        graph = self.get_collection(name_db, collection)

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

        if g1_candidates is not None:
            # Already present edges are not added again here:
            g1_connected = connect_graph_locally(g1_candidates,
                                                 distance_threshold)

            graph = self.get_collection(name_db,
                                        collection)
            
            edges = []
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
                    edges.append(edge)

            if edges:
                graph.insert_many(edges)
        
    def is_solved(self,
                  name_db,
                  collection,
                  x_lim,
                  y_lim,
                  z_lim):

        vertices, edges = self.__get_vertex_roi(name_db,
                                                collection,
                                                x_lim,
                                                y_lim,
                                                z_lim,
                                                query_edges=True)

        n_solved = len([e for e in edges if e["solved"]])
        n_edges = len(edges)

        return (n_edges == n_solved)

    
    def __get_vertex_roi(self,
                         name_db,
                         collection,
                         x_lim,
                         y_lim,
                         z_lim,
                         query_edges=True):

        logger.info("Get vertex ROI...")
        
        graph = self.get_collection(name_db, collection, overwrite=False)

        logger.info("Perform vertex query...")

        vertices =  list(graph.find({"$and": [{   
                                         "pz": {"$gte": z_lim["min"],
                                                 "$lt": z_lim["max"]},
                                         "py": {"$gte": y_lim["min"],
                                                 "$lt": y_lim["max"]},
                                         "px": {"$gte": x_lim["min"],
                                                 "$lt": x_lim["max"]}}, 
                                            {"type": "vertex"}]})
                         )

        vertex_ids = [v["id"] for v in vertices]
        
        edges = [] 
        if query_edges:
            logger.info("Perform edge query...")
            edges = list(graph.find({"$and": [{"id0": {"$in": vertex_ids}},
                                              {"id1": {"$in": vertex_ids}}]}))
        
        if not vertices:
            logger.warning("Warning, requested region holds no vertices!")
        if not edges:
            logger.warning("Warning, requested region holds no edges!")

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
          
            try:
                if e["cost"] is not None:
                    g1.set_edge_cost(e_g1, float(e["cost"]))
            except KeyError:
                pass
        
            if e["selected"]:
                g1.select_edge(e_g1)

            if e["solved"]:
                g1.solve_edge(e_g1)
        
        return g1, index_map_inv
