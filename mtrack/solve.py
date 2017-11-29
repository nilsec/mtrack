import numpy as np
import os
import json
from timeit import default_timer as timer
from shutil import copyfile
from copy import deepcopy
import glob
import h5py
from pymongo import MongoClient
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
    g1_solution = g2_to_g1_solution(g2_solution, g1, g2, index_maps, z_correction=z_correction, chunk_shift=chunk_shift)
   
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


def g2_to_g1_solution(g2_solution, g1, g2, index_maps, voxel_size=[5.,5.,50.], z_correction=None, chunk_shift=np.array([0.,0.,0.])):
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
                    min_vertices=4,
                    combine_solutions=True,
                    z_correction=1):

    """
    Solve a volume constrained by a bounding box directly from 
    the probability maps. prob_map_stack needs to be of DirectionType
    and specify perpendicular and parallel prob map path. Bounding box
    can also only specify the sections e.g. [300, 400] will crop the
    sections from the prob maps in the following way:
    for slice in range(300, 400)
        stack.append(prob_map_stack[:,:,slice])

    That is slices 300 - 399 will be processed. Note that the first slice
    is indexed with 0. In order to align the predictions with a tracing/knossos
    where the first slice is indexed with 1 the output of the 
    pipeline will need to be z_corrected:
        solution[x,y,z+1] == tracing[x,y,z]

    Thus a bb of [0, 10] = [0,1,...,9] will correspond to a tracing in sections
    [1, 2, ..., 10].
    """

    bounding_box = deepcopy(bounding_box)
    if z_correction is not None:
        if z_correction != 1:
            raise Warning("z correction != 1. Make sure this is correct.")

        assert(len(bounding_box) == 2)
        assert(bounding_box[0] > 0)
        bounding_box[0] -= z_correction
        bounding_box[1] -= z_correction
        print "bounding box: ", bounding_box[0]
        print "z_correction: ", z_correction

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
    
    cc_list = g1_connected.get_components(min_vertices=min_vertices,
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
                 combine_solutions,
                 z_correction)


class Core(Chunk):
    def __init__(self, voxel_size):
        Chunk.__init__(self, voxel_size)
    
class CoreSolver(object):
    def __init__(self):
        self.path_chunk_graph = None

    def get_subgraph(self,
                     name_db,
                     x_lim,
                     y_lim,
                     z_lim):

        print "Extract subgraph..."
        
        client = MongoClient()
        db = client[name_db]
        graph = db.graph

        print "Perform vertex query..."

        vertices =  list(graph.find({"$and": 
                                        [
                                         {"px": {"$gte": x_lim["min"],
                                                "$lt": x_lim["max"]}},
                                         {"py": {"$gte": y_lim["min"],
                                                 "$lt": y_lim["max"]}},
                                         {"pz": {"$gte": z_lim["min"],
                                                 "$lt": z_lim["max"]}}
                                        ]
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
                                                        {"id_mongo_0": id_mongo},
                                                        {"id_mongo_1": id_mongo}

                                                    ]
                                                },
                                                {"$and":
                                                    [
                                                        {"id_mongo_0": {"$in": vertex_ids}},
                                                        {"id_mongo_1": {"$in": vertex_ids}}
                                                    ]
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
        
        partner = []
        n = 0
        for v in vertices:
            g1.set_position(n, np.array([v["px"], v["py"], v["pz"]]))
            g1.set_orientation(n, np.array([v["ox"], v["oy"], v["oz"]]))
            partner.append(v["partner"])
            
            index_map[v["id"]] = n
            n += 1

        index_map[-1] = -1 

        index_map_get = np.vectorize(index_map.get)        
        partner = np.array(partner)
        g1.set_partner(0,0, vals=index_map_get(partner))

        n = 0
        for e in edges:
            e0 = index_map[np.uint64(e["id_0"])]
            e1 = index_map[np.uint64(e["id_1"])]
            g1.add_edge(e0, e1)

        return g1

        

    def get_chunk_graph(self, 
                        prob_map_stack_chunk,
                        offset_chunk,
                        gs,
                        ps,
                        voxel_size,
                        distance_threshold):
        """
        Generic

        offset_chunk: [x,y,z]
        """


        candidates = extract_candidates(prob_map_stack_chunk,
                                        gs,
                                        ps,
                                        voxel_size,
                                        bounding_box=None,
                                        bs_output_dir=None,
                                        offset_pos=offset_chunk)

        g1 = candidates_to_g1(candidates, 
                              voxel_size)

        g1_connected = connect_graph_locally(g1,
                                             distance_threshold)

        return g1_connected


    def save_chunk_graph(self, chunk_graph, name_db, id_chunk, overwrite=False):
        """
        db/gt 
        """
        print "Update database..."

        client = MongoClient()

        if overwrite:
            print "Overwrite {} database...".format(name_db)
            client.drop_database(name_db)

        db = client[name_db]
        
        graph = db.graph

        vertex_positions = chunk_graph.get_position_array().T
        vertex_orientations = chunk_graph.get_orientation_array().T
        partner = chunk_graph.get_vertex_property("partner").a
        edges = chunk_graph.get_edge_array()
        

        template_vertex = {"px": None,
                           "py": None,
                           "pz": None,
                           "ox": None,
                           "oy": None,
                           "oz": None,
                           "partner": None,
                           "id": None,
                           "id_chunk": None,
                           "edges": [],
                           "on": True,
                           "solved": False}

        print "Insert vertices..."

        index_map = {}
        vertices = []
        vertex_id = 0
        for pos, ori, partner in zip(vertex_positions,
                                     vertex_orientations,
                                     partner):

            vertex = deepcopy(template_vertex)

            vertex["px"] = pos[0]
            vertex["py"] = pos[1]
            vertex["pz"] = pos[2]
            vertex["ox"] = ori[0]
            vertex["oy"] = ori[1]
            vertex["oz"] = ori[2]
            vertex["partner"] = partner
            vertex["id"] = vertex_id
            vertex["id_chunk"] = id_chunk
        
            id_mongo = graph.insert_one(vertex).inserted_id
            index_map[vertex_id] = id_mongo 
            vertex_id += 1


        template_edge = {"id_0": None,
                         "id_1": None,
                         "id_mongo_0": None,
                         "id_mongo_1": None,
                         "id_chunk": None,
                         "on": True,
                         "solved": False}

        print "Insert edges..."

        for edge_id in edges:
            edge_mongo = [index_map[edge_id[0]], index_map[edge_id[1]]]
            edge = deepcopy(template_edge)
            
            edge["id_0"] = str(edge_id[0]) # convert uint64 to str, faster & mdb doesn't accept 64 bit int
            edge["id_1"] = str(edge_id[1])
            edge["id_mongo_0"] = edge_mongo[0]
            edge["id_mongo_1"] = edge_mongo[1]
            edge["id_chunk"] = id_chunk

            pm_edge_id = graph.insert_one(edge).inserted_id
            graph.update_one({'_id': edge_mongo[0]}, {"$addToSet": {"edges": pm_edge_id}})


def solve_chunks(prob_map_slice_dir,
                 max_chunk_shape,
                 overlap,
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
                 min_vertices=4,
                 volume_offset=np.array([0,0,0]),
                 phys=False,
                 combine_solutions=True,
                 z_correction=None):

    print "Get volume dimensions..."
    z_slices = glob.glob(prob_map_slice_dir.par + "/*.h5")
    f = h5py.File(z_slices[0], 'r')
    data = f["exported_data"]
    volume_shape = (len(z_slices), data.shape[1], data.shape[0])

    print "Chunk volume..."
    chunker = Chunker(volume_shape, 
                      max_chunk_shape,
                      voxel_size,
                      overlap,
                      offset=volume_offset,
                      phys=phys)

    chunks = chunker.chunk()
    
    chunk_dir = output_dir + "pm_chunks"
    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir + "/parallel")
        os.makedirs(chunk_dir + "/perpendicular")


    print "Chunk parallel probability maps..."
    slices_to_chunks(input_dir=prob_map_slice_dir.par,
                     output_dir=chunk_dir + "/parallel",
                     chunks=chunks)
    
    print "Chunk perpendicular probability maps..."
    slices_to_chunks(input_dir=prob_map_slice_dir.perp,
                     output_dir=chunk_dir + "/perpendicular",
                     chunks=chunks)

    n_chunks = len(chunks)
    parallel_chunks = [chunk_dir + "/parallel/chunk_{}.h5".format(chunk_id) for chunk_id in range(n_chunks)]
    perpendicular_chunks = [chunk_dir + "/perpendicular/chunk_{}.h5".format(chunk_id) for chunk_id in range(n_chunks)]

    n = 0
    for par, perp in zip(parallel_chunks, perpendicular_chunks):
        print "Solve chunk {}/{}...".format(n + 1, n_chunks)
         
        prob_map_stack = DirectionType(perp,
                                       par)

        candidates = extract_candidates(prob_map_stack,
                                        gs,
                                        ps,
                                        voxel_size,
                                        bounding_box=None,
                                        bs_output_dir=None)

        g1 = candidates_to_g1(candidates, 
                              voxel_size)

        g1_connected = connect_graph_locally(g1,
                                             distance_threshold)
    
        cc_list = g1_connected.get_components(min_vertices=min_vertices,
                                              output_folder = output_dir + "cc{}/".format(n))

        chunk_limits = chunks[n].limits
        chunk_shift = np.array([limit[0] for limit in chunk_limits])

        solve_volume(output_dir + "cc{}/".format(n),
                     start_edge_prior,
                     distance_factor,
                     orientation_factor,
                     comb_angle_factor,
                     selection_cost,
                     time_limit,
                     output_dir + "solution{}/".format(n),
                     voxel_size,
                     combine_solutions,
                     z_correction,
                     chunk_shift=chunk_shift)

        n += 1
