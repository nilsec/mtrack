import numpy as np
from copy import deepcopy
from pymongo import MongoClient, ASCENDING

from mtrack.graphs import g1_graph
from mtrack.preprocessing import extract_candidates, connect_graph_locally
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
                                "degree": None,
                                "solved": False}


        self.template_edge = {"id_v0_global": None,
                              "id_v1_global": None,
                              "id_v0_mongo": None,
                              "id_v1_mongo": None,
                              "selected": False}

        self.template_core = {"core_id": None}


    def _get_client(self, name_db, collection="graph", overwrite=False):
        print "Get client..."

        client = MongoClient(connect=False)

        db = client[name_db]
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

    def _get_db(self, name_db):
        client = MongoClient()
        db = client[name_db]
        return db

    def _check_forced(self, g1):
        """
        Check that the number of forced egdes
        incident to any vertex is <= 2 for 
        a given g1 graph.
        """
        for v in g1.get_vertex_iterator():
            incident = g1.get_incident_edges(v)
            forced = [g1.get_edge_property("force", u=e.source(), v=e.target()) for e in incident]
            assert(sum(forced)<=2)

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
        # Init index map, global <-> local:
        index_map = {}
        index_map_inv = {}

        # Flag solved edges & vertices:
        g1.new_vertex_property("force", dtype="bool", value=False)
        g1.new_edge_property("force", dtype="bool", vals=False)       


        partner = []
        n = 0
        for v in vertices:
            g1.set_position(n, np.array([v["px"], v["py"], v["pz"]]))
            g1.set_orientation(n, np.array([v["ox"], v["oy"], v["oz"]]))
            if v["solved"]:
                g1.set_vertex_property("force", n, True)

            partner.append(v["id_partner_global"])
            
            index_map[v["id_global"]] = n
            index_map_inv[n] = [v["id_global"], v["_id"]]
            n += 1

        index_map[-1] = -1
        
        if set_partner:
            partner = [index_map[p] for p in partner]
            partner = np.array(partner)
            g1.set_partner(0,0, vals=partner)

        for e in edges:
            e0 = index_map[np.uint64(e["id_v0_global"])]
            e1 = index_map[np.uint64(e["id_v1_global"])]
            e = g1.add_edge(e0, e1)
            g1.set_edge_property("force", u=0, v=0, value=True, e=e)

        self._check_forced(g1)
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
                       voxel_size,
                       time_limit,
                       hcs=False,
                       backend="Gurobi"):

        subgraph_connected = connect_graph_locally(subgraph, distance_threshold, cores=True)
        self._check_forced(subgraph_connected)

        print "Solve connected subgraphs..."
        if hcs:
            subgraph_connected.new_edge_property("weight", 
                                                 "int", 
                                                 vals=np.ones(
                                                    subgraph_connected.get_number_of_edges()
                                                             )
                                                 )

            ccs = subgraph_connected.get_hcs(subgraph_connected, 
                                             remove_singletons=4)
        else:
            ccs = subgraph_connected.get_components(min_vertices=cc_min_vertices,
                                                    output_folder="./ccs/",
                                                    return_graphs=True)

        j = 0
        solutions = []
        for cc in ccs:
            cc.reindex_edges_save()
            self._check_forced(cc)

            cc_solution = solve(cc,
                                start_edge_prior,
                                distance_factor,
                                orientation_factor,
                                comb_angle_factor,
                                selection_cost,
                                time_limit,
                                output_dir=None,
                                voxel_size=None,
                                chunk_shift=np.array([0.,0.,0.]),
                                backend=backend)

            for v in cc_solution.get_vertex_iterator():
                assert(len(cc_solution.get_incident_edges(v)) <= 2)
 
            solutions.append(cc_solution)

            j += 1

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
        edge_array = solution.get_edge_array()
        if edge_array.size:
            # We have no duplicate edge entries:
            unique_edges = np.vstack({tuple(row) for row in np.delete(edge_array, 2, 1)})
            assert(len(unique_edges) == len(edge_array))

            # We have no branchings in database:
            for v in solution.get_vertex_iterator():
                assert(len(solution.get_incident_edges(v)) <= 2)

            print "Insert solved edges..."
            for sol_edge in solution.get_edge_iterator():
                edge_id = [index_map[sol_edge.source()], index_map[sol_edge.target()]]
                edge = deepcopy(self.template_edge)
            
                edge["id_v0_global"] = str(edge_id[0][0])
                edge["id_v1_global"] = str(edge_id[1][0])
                edge["id_v0_mongo"] = edge_id[0][1]
                edge["id_v1_mongo"] = edge_id[1][1]
            
                # Check if edge lies in limits
                # vedges_glob = graph.find({"_id": {"$in": [edge_id[0][1], edge_id[1][1]]}})

                pos_vedges_loc = [np.array(solution.get_position(sol_edge.source())), 
                                  np.array(solution.get_position(sol_edge.target()))]

                edge_pos = np.array([0.,0.,0.])
                inside = False
                for v in pos_vedges_loc:
                    edge_pos += v

                if (x_lim is None) or (y_lim is None) or (z_lim is None):
                    assert(x_lim is None)
                    assert(y_lim is None)
                    assert(z_lim is None)
                    inside = True

                else:
                    edge_pos /= 2.
                    # Note that we check the half open interval s.t. we have no overlap:
                    if np.all(edge_pos >= min_lim) and np.all(edge_pos < max_lim):
                        inside = True
                    else:
                        inside = False
                
                if inside:
                    graph.update_many({"_id": {"$in": [edge_id[0][1], edge_id[1][1]]}}, 
                                      {"$inc": {"degree": 1}, 
                                       "$set": {"id_partner_global": -1, "solved": True}},
                                      upsert=False)

                    # Check if edge is already in db
                    edges_there = graph.find({"$and": [{"id_v0_mongo": edge["id_v0_mongo"]},
                                                       {"id_v1_mongo": edge["id_v1_mongo"]}]}).count()
                    assert(edges_there == 0)
                    
                    graph.insert_one(edge)
                    if graph.find({"degree": {"$gte": 3}}).count() != 0:
                        solution.save("./graph_violation.gt")
                        print "Violating edge: ", edge
                        print "Edge position: ", edge_pos
                        print "Write limits: ", min_lim, max_lim
                        raise ValueError("Degree = 3")

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
                
        for candidate in candidates:
            pos_phys = np.array([candidate.position[j] * voxel_size[j] for j in range(3)])
            ori_phys = np.array([candidate.orientation[j] * voxel_size[j] for j in range(3)])
            partner = candidate.partner_identifier
            vertex_id = candidate.identifier
           
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

        return vertex_id
