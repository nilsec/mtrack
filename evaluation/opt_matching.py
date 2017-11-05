import pylp
import process_solution
import pdb
import numpy as np
from scipy.spatial import KDTree
import postprocessing
import graphs
from dda3 import DDA3
import os
from shutil import copyfile
import json

def is_arr_in_list(arr, arr_list):
    return next((True for elem in arr_list if elem is arr), False)

def merge_dicts(a, b):
    c = a.copy()
    c.update(b)
    return c

def interpolate_lines(rec_line_paths, gt_line_paths):
    gt_lines = interpolate_nodes(gt_line_paths)
    rec_lines = interpolate_nodes(rec_line_paths)

    return rec_lines, gt_lines

def interpolate_nodes(line_list, voxel_size=[5.,5.,50.]):
    lines = []

    for line in line_list:
        if isinstance(line, str):
            g1 = graphs.g1_graph.G1(0)
            g1.load(line)
        else:
            g1 = line
        

        inter_line = []

        #Identify start and end node:
        start_end = []
        start_end_norm = []
        for v in g1.get_vertex_iterator():
            if len(g1.get_incident_edges(v)) == 1:
                start_end.append(v)
                start_end_norm.append(np.linalg.norm(g1.get_position(v) * np.array(voxel_size)))

        #Define the node with smaller norm as start node:
        if g1.get_number_of_vertices() > 1:
            if start_end_norm[0] < start_end_norm[1]:
                start_node = start_end[0]
                end_node = start_end[1]
            else:
                start_node = start_end[1]
                end_node = start_end[0]
        else:
            start_node = v
            end_node = v
            
        edge_start = start_node
        reached_end = False
        visited = []
        while not reached_end:
            neighbours = g1.get_neighbour_nodes(edge_start)
            if len(neighbours) > 0:
                if edge_start != start_node:
                    if len(neighbours) != 1:
                        edge_end_id = int(not neighbours.index(visited[-1]))
                        edge_end = neighbours[edge_end_id]
                    else:
                        edge_end_id = neighbours[0]
                        reached_end = True
                
                else: #Start/End nodes only have 1 neighbour
                    edge_end = neighbours[0]

                start = np.array(g1.get_position(edge_start), dtype=int)
                end = np.array(g1.get_position(edge_end), dtype=int)

                dda = DDA3(start, end)
                edge = dda.draw()

                inter_line.extend(edge) # Be careful here
                                        # the kdtree is scaled wih voxel size
                                        # 5,5,50, i.e. we need (x,y,z)

                visited.append(edge_start)
                edge_start = edge_end
            else:
                inter_line.extend([np.array(g1.get_position(edge_start), dtype=int)])
                reached_end = True
            
        lines.append(inter_line)

    return lines


class OptMatch(object):
    def __init__(self, lines_gt, lines_rec, n, distance_tolerance, dummy_cost, 
                 pair_cost_factor, max_edges, edge_selection_cost, voxel_size=np.array([5.0,5.0,50.0])):
        assert(isinstance(lines_gt, list))
        assert(isinstance(lines_rec, list))
    
        self.matching_parameters = {"chunk_size": n, 
                                    "max_edges": max_edges, 
                                    "edge_selection_cost": edge_selection_cost,
                                    "pair_cost_factor": pair_cost_factor,
                                    "dummy_cost": dummy_cost}

        print "Start optimal matching...\n"

        self.lines_gt = lines_gt
        self.lines_rec = lines_rec
        self.n = n

        print "Get tracing/ground truth chunks with size {}...\n".format(self.n)
        self.gt_chunks, self.gt_chunk_positions, self.inv_gt_chunk_positions, n_lines\
            = self.get_chunks(self.lines_gt, self.n)

        print "Get reconstruction chunks with size {}...\n".format(self.n)
        self.rec_chunks, self.rec_chunk_positions, self.inv_rec_chunk_positions, _\
            = self.get_chunks(self.lines_rec, self.n, l_0=n_lines)

        chunk_positions = self.rec_chunk_positions.values() + self.gt_chunk_positions.values()

        print "Connect chunks to each other...\n"
        self.pairs = self.connect_chunks(chunk_positions,
                                         voxel_size, 
                                         distance_tolerance)


        print "Find rec/trace chunk pairs...\n"
        self.gt_rec_pairs = []

        for pair in self.pairs:
            pos_0 = chunk_positions[pair[0]] 
            pos_1 = chunk_positions[pair[1]] 
            if is_arr_in_list(pos_0, self.rec_chunk_positions.values()):
                if is_arr_in_list(pos_1, self.gt_chunk_positions.values()):
                    self.gt_rec_pairs.append((self.inv_rec_chunk_positions[tuple(pos_0)],
                                              self.inv_gt_chunk_positions[tuple(pos_1)]))

            else:
                if is_arr_in_list(pos_1, self.rec_chunk_positions.values()):
                    self.gt_rec_pairs.append((self.inv_rec_chunk_positions[tuple(pos_1)],
                                              self.inv_gt_chunk_positions[tuple(pos_0)]))

        print "Get edge pairs...\n"
        self.edge_pairs, self.edge_pair_cost = self.get_edge_pairs(self.gt_rec_pairs)

        print "Initialize ILP...\n"
        self.backend = pylp.GurobiBackend()

        self.n_rec_nodes = len(self.rec_chunks.values())
        self.n_gt_nodes = len(self.gt_chunks.values())
        self.n_gt_rec_edges = len(self.gt_rec_pairs)
        self.n_edge_pairs = len(self.edge_pairs)
        self.n_dummy_edges = self.n_rec_nodes + self.n_gt_nodes

        self.n_tot = self.n_rec_nodes + self.n_gt_nodes +\
                     self.n_gt_rec_edges + self.n_edge_pairs +\
                     self.n_dummy_edges

        self.backend.initialize(self.n_tot, 
                                pylp.VariableType.Binary)
        
        pylp.setLogLevel()

        print "Set costs...\n"
        # Costs
        self.objective = pylp.LinearObjective(self.n_tot)
        
        start_id = 0
        self.rec_node_start_id = start_id
        for rec_node_id in range(start_id, self.n_rec_nodes):
            self.objective.set_coefficient(rec_node_id,
                                           0.0)

        start_id += self.n_rec_nodes
        self.gt_node_start_id = start_id
        for gt_node_id in range(start_id, start_id + self.n_gt_nodes):
            self.objective.set_coefficient(gt_node_id,
                                           0.0)

        start_id += self.n_gt_nodes
        self.edge_start_id = start_id
        for edge_id in range(start_id, start_id + self.n_gt_rec_edges):
            self.objective.set_coefficient(edge_id,
                                           edge_selection_cost)

        start_id += self.n_gt_rec_edges
        self.edge_pair_start_id = start_id
        for edge_pair_id in range(start_id, start_id + self.n_edge_pairs):
            self.objective.set_coefficient(edge_pair_id,
                                           self.edge_pair_cost[edge_pair_id - start_id] * pair_cost_factor)

        start_id += self.n_edge_pairs
        self.dummy_start_id = start_id
        for dummy_edge_id in range(start_id, start_id + self.n_dummy_edges):
            self.objective.set_coefficient(dummy_edge_id,
                                           dummy_cost)

        self.node_variables = dict(zip(range(self.rec_node_start_id, self.edge_start_id),\
                                  self.rec_chunks.keys() + self.gt_chunks.keys()))

        self.edge_variables = dict(zip(range(self.edge_start_id, self.edge_pair_start_id),\
                                  self.gt_rec_pairs))

        self.edge_pair_variables = dict(zip(range(self.edge_pair_start_id, self.dummy_start_id),\
                                       self.edge_pairs))

        self.dummy_variables = dict(zip(range(self.dummy_start_id, self.dummy_start_id + self.n_dummy_edges),
                                   zip(["d"] * self.n_dummy_edges, self.rec_chunks.keys() + self.gt_chunks.keys())))
            

        variable_list = [self.node_variables, 
                         self.edge_variables, 
                         self.edge_pair_variables, 
                         self.dummy_variables]

        self.variables = {}
        for v in variable_list:
            self.variables = merge_dicts(self.variables, v)

        self.backend.set_objective(self.objective)

        print "Find constraints and conflicts...\n"
        # Conflicts & Constraints
        self.constraints = pylp.LinearConstraints()

        edge_pair_constraints = self.get_edge_pair_constraints()
        edge_constraints = self.get_edge_constraints()
        node_edges = self.get_edges_to_node()
        dummy_constraints = self.get_dummy_constraints()
        cross_conflicts = self.get_cross_conflicts()
        id_conflicts = self.get_id_conflicts()
        
        for node in range(self.n_gt_nodes):
            constraint = pylp.LinearConstraint()
            constraint.set_coefficient(node, 1)
            constraint.set_relation(pylp.Relation.Equal)
            constraint.set_value(1)

            #self.constraints.add(constraint)

        for edge_pair in edge_pair_constraints:
            # Pair implies edges
            constraint = pylp.LinearConstraint()

            constraint.set_coefficient(edge_pair[0], 2) # pair
            constraint.set_coefficient(edge_pair[1], -1)
            constraint.set_coefficient(edge_pair[2], -1)
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(0)

            self.constraints.add(constraint)

            # Edges imply pair
            constraint = pylp.LinearConstraint()

            constraint.set_coefficient(edge_pair[1], 1)
            constraint.set_coefficient(edge_pair[2], 1)
            constraint.set_coefficient(edge_pair[0], -1) # pair
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(1)
            
            self.constraints.add(constraint)

        for edge_nodes in edge_constraints:
            # Edge implies nodes
            constraint = pylp.LinearConstraint()
            
            constraint.set_coefficient(edge_nodes[0], 2)
            constraint.set_coefficient(edge_nodes[1], -1)
            constraint.set_coefficient(edge_nodes[2], -1)
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(0)
    
            self.constraints.add(constraint)

        for edges_to_node in node_edges:
            # node_edges = [(node_id, edge_to_node_1, edge_to_node_2, ...), ...]

            # If the dummy node is chosen no other edge can be chosen
            constraint = pylp.LinearConstraint()

            for edge in edges_to_node[1:]:
                constraint.set_coefficient(edge, 1)

            #dummy node corresponding to node_i:
            dummy = edges_to_node[0] + self.dummy_start_id
            constraint.set_coefficient(dummy, max_edges)
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(max_edges)
            
            self.constraints.add(constraint)

            # Every node needs to be explained by dummy or edge
            constraint = pylp.LinearConstraint()

            for edge in edges_to_node[1:]:
                constraint.set_coefficient(edge, 1)
            
            constraint.set_coefficient(dummy, max_edges)
            constraint.set_relation(pylp.Relation.GreaterEqual)
            constraint.set_value(1)

            self.constraints.add(constraint)

        for dummy_node in dummy_constraints:
            # Dummy implies node:
            constraint = pylp.LinearConstraint()
        
            constraint.set_coefficient(dummy_node[0], 1)
            constraint.set_coefficient(dummy_node[1], -1)
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(0)

            self.constraints.add(constraint)

        for cross_edges in cross_conflicts:
            # Crossing edges are exclusive
            assert(len(cross_edges) == 2)
            constraint = pylp.LinearConstraint()
            
            for edge in cross_edges:
                constraint.set_coefficient(edge, 1)

            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(1)
            
            self.constraints.add(constraint)

        for id_conflict in id_conflicts:
            # Each node can only have edges that point to same line id
            assert(len(id_conflict) == 2)
            constraint = pylp.LinearConstraint()
        
            for edge in id_conflict:
                constraint.set_coefficient(edge, 1)

            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(1)

            self.constraints.add(constraint)

        print "Set {} constraints...\n".format(len(self.constraints))
        self.backend.set_constraints(self.constraints)

    def solve(self, time_limit=None):
        print "Solve ILP with: " + str(len(self.constraints)) + " constraints and "
        print str(self.n_tot) + " variables.\n"

        if time_limit != None:
            self.backend.set_timelimit(time_limit)

        solution = pylp.Solution()
        self.backend.solve(solution)
       
        non_zero = 0
        for i in range(len(solution)):
            if solution[i] > 0.5:
                non_zero += 1

        print "Solution found with {} non zero variables.\n".format(non_zero)

        return solution

    def __match_solution_lines(self, lines, chunk_keys, chunk_matches, start_id, dummys):
        line_matches = {line_id: [] for line_id in range(start_id, len(lines) + start_id)}
        stats = {"matches": 0, "dummys": 0, "switches": 0}

        for line_id in line_matches.keys():
            chunks = [[line_id, None] for x in chunk_keys if x[0] == line_id]

            chunk_id = 0

            for chunk in chunks:
                chunk[1] = chunk_id

                try:
                    match_idx = [x[0] for x in chunk_matches].index(tuple(chunk))
                    line_matches[line_id].append(chunk_matches[match_idx][1])
                    stats["matches"] += 1

                except ValueError:
                    assert(("d", tuple(chunk)) in dummys)
                    line_matches[line_id].append("d")
                    stats["dummys"] += 1
           
                try: 
                    before = line_matches[line_id][-2][0]
                    after = line_matches[line_id][-1][0] 
                    if before != after:
                        if before != "d" and after != "d":
                            stats["switches"] += 1
                except IndexError:
                    pass

                chunk_id += 1

    
            # Count switches (mergers splits) but we do not
            # want to include switches from dummy to line
            # but the number of switches should be at least
            # equal to the number of unique line ids
            # import in situation like (1, d, 2)
            # which is not captured by the try except block above
            unique_ids = set([x[0] for x in line_matches[line_id]])
            try:
                unique_ids.remove("d")
            except KeyError:
                pass

            stats["switches"] = max(stats["switches"], len(unique_ids) - 1)
 

        return line_matches, stats
 

    def evaluate_solution(self, solution, gt_line_path_list, rec_line_path_list, base_save_folder):
        mergers = 0
        splits = 0
        
        tp_nodes = 0
        fn_nodes = 0
        fp_nodes = 0


        selected_edges = [self.variables[i] for i in range(self.edge_start_id, self.edge_pair_start_id)\
                          if solution[i] > 0.5]


        selected_dummys = [self.variables[i] for i in range(self.dummy_start_id, 
                                                            self.dummy_start_id + self.n_dummy_edges)\
                                                            if solution[i] > 0.5]


        matches_rec_gt = sorted(selected_edges, key=lambda x: (x[0][1], x[0][0])) #(rec, gt)
        matches_1 = sorted(selected_edges, key=lambda x: (x[1][1], x[1][0])) 
        matches_gt_rec = [(x[1], x[0]) for x in matches_1] #(gt, rec)

        gt_line_matches = {gt_line_id: [] for gt_line_id in range(len(self.lines_gt))}
        rec_line_matches = {rec_line_id: [] for rec_line_id in range(len(self.lines_rec))}

        gt_line_matches, gt_stats = self.__match_solution_lines(self.lines_gt, self.gt_chunks.keys(), matches_gt_rec, 0, selected_dummys)
        rec_line_matches, rec_stats = self.__match_solution_lines(self.lines_rec, self.rec_chunks.keys(), matches_rec_gt, len(self.lines_gt), selected_dummys)

        tot_eval = {"tp": rec_stats["matches"], 
                    "fn": gt_stats["dummys"], 
                    "fp": rec_stats["dummys"],
                    "mergers": rec_stats["switches"],
                    "splits": gt_stats["switches"]}

        self.save_solution(tot_eval,
                           gt_line_matches,
                           rec_line_matches,
                           gt_line_path_list,
                           rec_line_path_list,
                           base_save_folder)


    def save_solution(self, tot_eval,
                            gt_line_matches, 
                            rec_line_matches, 
                            gt_line_path_list, 
                            rec_line_path_list,
                            base_save_folder):

        eval_number = 0
        base_save_folder += "_{}"
        while os.path.exists(base_save_folder.format(eval_number)):
            eval_number += 1

        base_save_folder = base_save_folder.format(eval_number)
            
            

        fn_folder = os.path.join(base_save_folder, "fn")
        fp_folder = os.path.join(base_save_folder, "fp")

        if not os.path.exists(fn_folder):
            os.makedirs(fn_folder)
        if not os.path.exists(fp_folder):
            os.makedirs(fp_folder)
        
        json.dump(tot_eval,
                  open(os.path.join(base_save_folder, "chunk_evaluation.json"), "w+"))

        line_evaluation = {"fn": 0,"fp": 0, "tp": 0, "splits": 0, "mergers": 0}
 
        for gt_line, match_list in gt_line_matches.iteritems():
            folder = os.path.join(base_save_folder, "gt/" + str(gt_line))
            if not os.path.exists(folder):
                os.makedirs(folder)

            dummy_chunks = 0
            matches = {}
            for match in match_list:
                if not (match == "d"):
                    rec_line_id = match[0] - len(self.lines_gt)

                    copyfile(gt_line_path_list[gt_line][:-3] + "_kfy.nml", 
                             os.path.join(folder, "BASE_gt" + str(gt_line) + ".nml"))
  
                    copyfile(rec_line_path_list[rec_line_id][:-3] + "_kfy.nml", 
                             os.path.join(folder, "MATCH_rec" + str(rec_line_id) + ".nml"))

                    try:
                        matches["rec" + str(rec_line_id)] += 1
                    except KeyError:
                        matches["rec" + str(rec_line_id)] = 1
                else:
                    dummy_chunks += 1

            matches["d"] = dummy_chunks

            #Check for full false negatives:
            if matches["d"] == len(match_list):
                os.rmdir(folder)
                line_evaluation["fn"] += 1

                copyfile(gt_line_path_list[gt_line][:-3] + "_kfy.nml", 
                         os.path.join(fn_folder, "gt" + str(gt_line) + ".nml"))
            else:
                json.dump(matches, 
                          open(os.path.join(folder, "STATS_gt" + str(gt_line) + ".json"), 
                               "w+"))

                # Count number of excess rec files in folder. -2 because of base gives splits
                splits = len([f for f in os.listdir(folder) if f.endswith(".nml")]) - 2
                line_evaluation["splits"] += splits

 
        #-------------------------------------------------


        for rec_line, match_list in rec_line_matches.iteritems():
            rec_line_id = rec_line - len(self.lines_gt)

            folder = os.path.join(base_save_folder, "rec/" + str(rec_line_id))
            if not os.path.exists(folder):
                os.makedirs(folder)

            dummy_chunks = 0
            matches = {}
            for match in match_list:
                if not (match == "d"):
                    gt_line_id = match[0]

                    copyfile(rec_line_path_list[rec_line_id][:-3] + "_kfy.nml", 
                          os.path.join(folder, "BASE_rec" + str(rec_line_id) + ".nml"))
  
                    copyfile(gt_line_path_list[gt_line_id][:-3] + "_kfy.nml", 
                          os.path.join(folder, "MATCH_gt" + str(gt_line_id) + ".nml"))

                    try:
                        matches["gt" + str(gt_line_id)] += 1
                    except KeyError:
                        matches["gt" + str(gt_line_id)] = 1

                else:
                    dummy_chunks += 1

            matches["d"] = dummy_chunks

            #Check for full false positives:
            if matches["d"] == len(match_list):
                os.rmdir(folder)
                line_evaluation["fp"] += 1

                copyfile(rec_line_path_list[rec_line_id][:-3] + "_kfy.nml", 
                         os.path.join(fp_folder, "rec" + str(rec_line_id) + ".nml"))
            else:
                json.dump(matches, 
                          open(os.path.join(folder, "STATS_rec" + str(rec_line_id) + ".json"), 
                               "w+"))
                line_evaluation["tp"] += 1

                # Count number of excess gt files in folder. -2 because of base gives splits
                mergers = len([f for f in os.listdir(folder) if f.endswith(".nml")]) - 2
                line_evaluation["mergers"] += mergers
 

        json.dump(line_evaluation,
                  open(os.path.join(base_save_folder, "line_evaluation.json"), "w+"))

        json.dump(self.matching_parameters,
                  open(os.path.join(base_save_folder, "matching_params.json"), "w+"))
 
 
                
        
        
    def get_edge_pair_constraints(self):
        constraints = [] # constraints will be tuple of the form (edge_pair_id, edge_id_1, edge_id_2)
        
        edge_pair_id = 0
        for edge_pair in self.edge_pairs:
            edge_id_1 = self.gt_rec_pairs.index(edge_pair[0])
            edge_id_2 = self.gt_rec_pairs.index(edge_pair[1])
            constraints.append((edge_pair_id + self.edge_pair_start_id, 
                                edge_id_1 + self.edge_start_id, 
                                edge_id_2 + self.edge_start_id))
            edge_pair_id += 1

        return constraints

    def get_edge_constraints(self):
        constraints = [] # constraints will be tuple of the form (edge_id, node_id_1, node_id_2)
        node_list = self.rec_chunks.keys() + self.gt_chunks.keys()

        edge_id = 0
        for edge in self.gt_rec_pairs:
            node_id_1 = node_list.index(edge[0])
            node_id_2 = node_list.index(edge[1])
            constraints.append((edge_id + self.edge_start_id, 
                                node_id_1 + self.rec_node_start_id, 
                                node_id_2 + self.rec_node_start_id))
            edge_id += 1

        return constraints

    def get_dummy_constraints(self):
        # dummy - node <= 0 i.e. dummy implies node
        # constraints = [(dummy, node), ...]
        node_list = self.rec_chunks.keys() + self.gt_chunks.keys()

        constraints = [(i + self.dummy_start_id, i + self.rec_node_start_id) for i in range(len(node_list))]
        
        return constraints

    def get_cross_conflicts(self):
        # We have exclusive edges that cross:
        # An edge is in conflict with all other edges for which:
        # e_1[0][1] > e_2[0][1] and e_1[1][1] < e_2[1][1]

        conflicts = []
        for i in range(len(self.gt_rec_pairs)):
            edge_i = self.gt_rec_pairs[i]
            c_i = set([i + self.edge_start_id])

            for j in range(len(self.gt_rec_pairs)):

                edge_j = self.gt_rec_pairs[j]
                if edge_i[0][0] == edge_j[0][0]:
                    if edge_i[1][0] == edge_j[1][0]:

                        if edge_i[0][1] > edge_j[0][1]:
                            if edge_i[1][1] < edge_j[1][1]:
                                c_i.add(j + self.edge_start_id)
                                conflicts.append(tuple(c_i))

                                c_i = set([i + self.edge_start_id])

        return conflicts
                    
    def get_edges_to_node(self):
        node_list = self.rec_chunks.keys() + self.gt_chunks.keys()
        
        node_edges = [] # [(node_id, edge_to_node_1, edge_to_node_2, ...), ...]        
        node_id = 0
        for node in node_list:

            edges_to_node = []
            edge_id = 0
            for edge in self.gt_rec_pairs:
                if node in edge:
                    edges_to_node.append(edge_id + self.edge_start_id)

                edge_id += 1

            tmp = [node_id + self.rec_node_start_id] + edges_to_node
            node_edges.append(tuple(tmp))

            node_id += 1

        return node_edges

    def get_id_conflicts(self):
        conflicts = []
        edges_to_node = self.get_edges_to_node()
        node_list = self.rec_chunks.keys() + self.gt_chunks.keys()

        for node_edges in edges_to_node:
            for i in range(1, len(node_edges)):
                for j in range(i + 1, len(node_edges)):
                   
                    node = node_list[node_edges[0] - self.rec_node_start_id] 
                    edge_i = self.gt_rec_pairs[node_edges[i] - self.edge_start_id]
                    edge_j = self.gt_rec_pairs[node_edges[j] - self.edge_start_id]

                    node_id_i = edge_i.index(node)
                    node_id_j = edge_j.index(node)

                    if edge_i[int(not node_id_i)][0] != edge_j[int(not node_id_j)][0]:
                        conflicts.append((node_edges[i], node_edges[j]))
                        

        return conflicts

    def get_chunks(self, lines, n, l_0=0):
        chunks = {}
        chunk_positions = {}
        inv_chunk_positions = {}

        l = l_0
        for line in lines:

            k = 0
            chunks[(l, k)] = []
            pos = np.array([0,0,0])

            m = 0
            for voxel in line:
                chunks[(l, k)].append(voxel)
                pos += voxel
                m += 1

                if not m % n:
                    chunk_pos = pos/float(n)
                    chunk_positions[(l, k)] = chunk_pos
                    inv_chunk_positions[tuple(chunk_pos)] = (l,k)
                    k += 1
                    chunks[l,k] = []
                    pos = np.array([0,0,0])

                if m == len(line) and m % n:
                    chunk_pos = pos/(m%n)

                    chunk_positions[(l, k)] = chunk_pos
                    inv_chunk_positions[tuple(chunk_pos)] = (l,k)
            l += 1
 
        return chunks, chunk_positions, inv_chunk_positions, l

    def connect_chunks(self, chunk_positions, voxel_size, distance_tolerance):
        kdtree = KDTree(chunk_positions * voxel_size)
        pairs = kdtree.query_pairs(distance_tolerance, p=2.0, eps=0)
    
        partner_list = []
        for j in range(len(chunk_positions)):
            partner = False
            for i in range(len(chunk_positions)):
                d = np.linalg.norm((chunk_positions[j] - chunk_positions[i]) * voxel_size) 
                if d < distance_tolerance:
                    if i != j:
                        partner=True
            partner_list.append(partner)

        return pairs

    def get_edge_pairs(self, gt_rec_pairs):
        N_edges = len(gt_rec_pairs)
        edge_pairs = []
        edge_pair_cost = []

        for i in range(N_edges):
            for j in range(N_edges):
                if j>i:
                    edge_i = gt_rec_pairs[i]
                    edge_j = gt_rec_pairs[j]

                    edge_i_lines = [edge_i[0][0], edge_i[1][0]]
                    edge_j_lines = [edge_j[0][0], edge_j[1][0]]

                    match = None
                    line_i = 0
                    for i_id in edge_i_lines:
                        line_j = 0
                        for j_id in edge_j_lines:
                            if i_id == j_id:
                                match = (line_i, line_j)
                            line_j += 1
                        line_i += 1
                                
                    if match is not None:
                        if abs(edge_i[match[0]][1] - edge_j[match[1]][1]) == 1:
                            edge_pairs.append((edge_i, edge_j))
                            if edge_i[int(not match[0])][0] != edge_j[int(not match[1])][0]:
                                edge_pair_cost.append(1)
                            else:
                                edge_pair_cost.append(0)
            
        return edge_pairs, edge_pair_cost
