import numpy as np
from mtrack.graphs.g1_graph import G1
import pylp

class G1Solver:
    def __init__(self, 
                 g1, 
                 distance_factor, 
                 orientation_factor,
                 start_edge_prior,
                 comb_angle_factor,
                 vertex_selection_cost,
                 backend="Gurobi"):

        if backend == "Gurobi":
            self.backend = pylp.GurobiBackend()
        elif backend == "Scip":
            self.backend = pylp.ScipBackend()
        else:
            raise NotImplementedError("Choose between Gurobi or Scip backend")

        self.g1 = g1

        self.distance_factor = distance_factor
        self.orientation_factor = orientation_factor
        self.start_edge_prior = start_edge_prior
        self.comb_angle_factor = comb_angle_factor
        self.vertex_selection_cost = vertex_selection_cost

        self.vertex_cost = g1.get_vertex_cost()

        self.edge_cost = g1.get_edge_cost(distance_factor,
                                          orientation_factor,
                                          start_edge_prior)

        self.edge_combination_cost, self.edges_to_middle =\
            g1.get_edge_combination_cost(comb_angle_factor=comb_angle_factor, 
                                         return_edges_to_middle=True)

        self.n_vertices = g1.get_number_of_vertices()
        self.n_dummy = self.n_vertices
        self.n_edges = g1.get_number_of_edges() + self.n_dummy
        self.n_comb_edges = len(self.edge_combination_cost)

        # Variables are vertices, edges and combination of egdes
        self.n_variables = self.n_vertices + self.n_edges + self.n_comb_edges
         
        self.backend.initialize(self.n_variables, pylp.VariableType.Binary)
        
        self.objective = pylp.LinearObjective(self.n_variables)
        
        """
        Set costs
        """
        
        binary_id = 0
        # Add one variable for each vertex (selection cost only for mt's)
        self.vertex_to_binary = {}
        self.binary_to_vertex = {}
        for v in g1.get_vertex_iterator():
            self.objective.set_coefficient(binary_id, 
                                           self.vertex_selection_cost +\
                                           self.vertex_cost[v])

            self.vertex_to_binary[v] = binary_id 
            self.binary_to_vertex[binary_id] = v
            binary_id += 1

        assert(binary_id == self.n_vertices)

        # Add one variable for each edge
        self.edge_to_binary = {}
        self.binary_to_edge = {}
        for e in g1.get_edge_iterator():
            self.objective.set_coefficient(binary_id,
                                           self.edge_cost[e])
            self.edge_to_binary[e] = binary_id
            self.binary_to_edge[binary_id] = e
            binary_id += 1

        # Add one dummy edge for each vertex
        self.dummy_to_binary = {}
        self.binary_to_dummy = {}
        for v in g1.get_vertex_iterator():
            self.objective.set_coefficient(binary_id,
                                           self.edge_cost[G1.START_EDGE])

            self.dummy_to_binary[v] = binary_id
            self.binary_to_dummy[binary_id] = v
            binary_id += 1
        assert(binary_id == self.n_vertices + self.n_edges)

        # Add one variable for each combination of edges:
        self.comb_to_binary = {}
        self.binary_to_comb = {}
        for ee, cost in self.edge_combination_cost.iteritems():
            self.objective.set_coefficient(binary_id,
                                           cost)

            self.comb_to_binary[ee] = binary_id
            self.binary_to_comb[binary_id] = ee
            binary_id += 1
        assert(binary_id == self.n_variables)
       
        self.backend.set_objective(self.objective)

        """
        Constraints
        """ 
        self.constraints = pylp.LinearConstraints()

        # Edge selection implies vertex selection:
        for e in g1.get_edge_iterator():
            v0 = e.source()
            v1 = e.target()
           
            constraint = pylp.LinearConstraint()
            constraint.set_coefficient(self.edge_to_binary[e], 2)
            constraint.set_coefficient(self.vertex_to_binary[v0], -1)
            constraint.set_coefficient(self.vertex_to_binary[v1], -1)
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(0)
            
            self.constraints.add(constraint)

        # Vertex selection implies 2 edges:
        for v in g1.get_vertex_iterator():
            incident_edges = g1.get_incident_edges(v)
            
            constraint = pylp.LinearConstraint()
            constraint.set_coefficient(self.vertex_to_binary[v], 2)

            constraint.set_coefficient(self.dummy_to_binary[v], -1)
            for e in incident_edges:
                constraint.set_coefficient(self.edge_to_binary[e], -1)

            constraint.set_relation(pylp.Relation.Equal)
            constraint.set_value(0)

            self.constraints.add(constraint)

        # Combination of 2 edges implies edges and vice versa:
        for ee in self.edge_combination_cost.keys():
            e0 = ee[0]
            e1 = ee[1]
            
            assert(e0 != G1.START_EDGE or e1 != G1.START_EDGE)

            if e0 == G1.START_EDGE:
                middle_vertex = self.edges_to_middle[ee]
                b0 = self.dummy_to_binary[middle_vertex]
                b1 = self.edge_to_binary[e1]

            elif e1 == G1.START_EDGE:
                middle_vertex = self.edges_to_middle[ee]
                b0 = self.edge_to_binary[e0]
                b1 = self.dummy_to_binary[middle_vertex]

            else:
                b0 = self.edge_to_binary[e0]
                b1 = self.edge_to_binary[e1]            

            constraint = pylp.LinearConstraint()
            constraint.set_coefficient(self.comb_to_binary[ee], 2)
            constraint.set_coefficient(b0, -1)
            constraint.set_coefficient(b1, -1)
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(0)
            
            self.constraints.add(constraint)

            # Edges implies combination:
            constraint = pylp.LinearConstraint()
            constraint.set_coefficient(b0, 1)
            constraint.set_coefficient(b1, 1)
            constraint.set_coefficient(self.comb_to_binary[ee], -1)
            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(1)
            
            self.constraints.add(constraint)

        # Add partner constraints:
        for v in g1.get_vertex_iterator():
            partner = g1.get_partner(v)
            if partner != -1:
                if v < partner:
                    constraint = pylp.LinearConstraint()
                    constraint.set_coefficient(self.vertex_to_binary(v), 1)
                    constraint.set_coefficient(self.vertex_to_binary(partner), 1)
                    constraint.set_relation(pylp.Relation.LessEqual)
                    constraint.set_value(1)
                    self.constraints.add(constraint)


    def solve(self, time_limit=None):
        print "with" + str(len(self.constraints)) + "constraints."
        print "and " + str(self.n_variables) + " variables.\n"

        if time_limit != None:
            try:
                self.backend.set_timelimit(time_limit)
            except AttributeError:
                print "WARNING: Unable to set time limit"
                pass

        solution = pylp.Solution()
        self.backend.solve(solution)
        
        return solution

    def solution_to_g1(self, 
                       solution,
                       voxel_size,
                       chunk_shift=np.array([0.,0.,0.])):

        vertex_mask = []
        edge_mask = []

        for v in self.g1.get_vertex_iterator():
            if solution[self.vertex_to_binary[v]] > 0.5:
                vertex_mask.append(True)
                pos = self.g1.get_position(v)
                pos += np.array(chunk_shift) * np.array(voxel_size)
                self.g1.set_position(v, np.array(pos))
            else:
                vertex_mask.append(False)

        for e in self.g1.get_edge_iterator():
            if solution[self.edge_to_binary[e]] > 0.5:
                edge_mask.append(True)
            else:
                edge_mask.append(False)

        self.g1.g.set_vertex_filter(None)
        self.g1.g.set_edge_filter(None)

        vertex_filter = self.g1.g.new_vertex_property("bool")
        edge_filter = self.g1.g.new_edge_property("bool")

        vertex_filter.a = vertex_mask
        edge_filter.a = edge_mask

        self.g1.g.set_vertex_filter(vertex_filter)
        self.g1.g.set_edge_filter(edge_filter)

        return self.g1
