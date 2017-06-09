import pylp
import numpy as np

class G2Solver:
    def __init__(self, g2):
        self.g2_vertices_N = g2.get_number_of_vertices()
        
        self.backend = pylp.GurobiBackend()
        self.backend.initialize(self.g2_vertices_N, pylp.VariableType.Binary)
        self.objective = pylp.LinearObjective(self.g2_vertices_N)

        g2_vertex_index_map = g2.get_vertex_index_map()

        for v in g2.get_vertex_iterator():
            self.objective.set_coefficient(g2_vertex_index_map[v], 
                                           g2.get_cost(v))

        self.backend.set_objective(self.objective)
        self.constraints = pylp.LinearConstraints()

        j = 0
        for conflict in g2.get_conflicts():

            constraint = pylp.LinearConstraint()

            for v in conflict:
                constraint.set_coefficient(v, 1)

            constraint.set_relation(pylp.Relation.LessEqual)
            constraint.set_value(1)
            self.constraints.add(constraint)
            j += 1 

        for sum_constraint in g2.get_sum_constraints():

            vertices_1 = sum_constraint[0]
            vertices_2 = sum_constraint[1]

            constraint = pylp.LinearConstraint()

            for v in vertices_1:
                constraint.set_coefficient(v, 1)
            for v in vertices_2:
                constraint.set_coefficient(v, -1)

            constraint.set_relation(pylp.Relation.Equal)
            constraint.set_value(0)
            self.constraints.add(constraint)

        self.backend.set_constraints(self.constraints)

    def solve(self, time_limit=None):
        print "with: " + str(len(self.constraints)) + " constraints."
        print "and " + str(self.g2_vertices_N) + " variables.\n"

        if time_limit != None:
            self.backend.set_timelimit(time_limit)

        solution = pylp.Solution()
        self.backend.solve(solution)

        return solution
