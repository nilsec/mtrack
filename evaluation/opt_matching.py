import pylp
import pdb
import numpy as np
from scipy.spatial import KDTree

def is_arr_in_list(arr, arr_list):
    return next((True for elem in arr_list if elem is arr), False)

def merge_dicts(a, b):
    c = a.copy()
    c.update(b)
    return c

class OptMatch(object):
    def __init__(self, lines_gt, lines_rec, n, distance_tolerance, dummy_cost, 
                 pair_cost_factor, max_edges, edge_selection_cost, voxel_size=np.array([5.0,5.0,50.0])):
        assert(isinstance(lines_gt, list))
        assert(isinstance(lines_rec, list))

        self.lines_gt = lines_gt
        self.lines_rec = lines_rec
        self.n = n

        self.gt_chunks, self.gt_chunk_positions, self.inv_gt_chunk_positions, n_lines\
            = self.get_chunks(self.lines_gt, self.n)

        self.rec_chunks, self.rec_chunk_positions, self.inv_rec_chunk_positions, _\
            = self.get_chunks(self.lines_rec, self.n, l_0=n_lines)

        chunk_positions = self.rec_chunk_positions.values() + self.gt_chunk_positions.values() 

        self.pairs = self.connect_chunks(chunk_positions,
                                         voxel_size, 
                                         distance_tolerance)

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

        self.edge_pairs, self.edge_pair_cost = self.get_edge_pairs(self.gt_rec_pairs)

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

    def evaluate_solution(self, solution):
        mergers = 0
        splits = 0
        
        tp_nodes = 0
        fn_nodes = 0
        fp_nodes = 0

        selected_nodes = [self.variables[i] for i in range(0, self.edge_start_id) if solution[i] > 0.5]

        selected_edges = [self.variables[i] for i in range(self.edge_start_id, self.edge_pair_start_id)\
                          if solution[i] > 0.5]

        selected_edge_pairs = [self.variables[i] for i in range(self.edge_pair_start_id, self.dummy_start_id)\
                               if solution[i] > 0.5]

        selected_dummys = [self.variables[i] for i in range(self.dummy_start_id, 
                                                            self.dummy_start_id + self.n_dummy_edges)\
                                                            if solution[i] > 0.5]

        matches_0 = sorted(selected_edges, key=lambda x: (x[0][1], x[0][0]))
        matches_1 = sorted(selected_edges, key=lambda x: (x[1][1], x[1][0]))

        print matches_0
        print matches_1
        print selected_dummys

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
