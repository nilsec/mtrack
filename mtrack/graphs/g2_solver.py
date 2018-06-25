import numpy as np
import pylp

class G2Solver:
    def __init__(self, g2, backend="Gurobi"):
        self.g2_vertices_N = g2.get_number_of_vertices()
        
        if backend == "Gurobi":
            self.backend = pylp.GurobiBackend()
        elif backend == "Scip":
            self.backend = pylp.ScipBackend()
        else:
            raise NotImplementedError("Choose between Gurobi or Scip backend")

        self.backend.initialize(self.g2_vertices_N, pylp.VariableType.Binary)

        self.objective = pylp.LinearObjective(self.g2_vertices_N)

        pylp.setLogLevel()

        g2_vertex_index_map = g2.get_vertex_index_map()
        self.constraints = pylp.LinearConstraints()

        for v in g2.get_vertex_iterator():
            self.objective.set_coefficient(g2_vertex_index_map[v], 
                                           g2.get_cost(v))

            constraint = pylp.LinearConstraint()
            if g2.get_forced(v):
                constraint.set_coefficient(v, 1)
                constraint.set_relation(pylp.Relation.Equal)
                constraint.set_value(1)
                self.constraints.add(constraint)

        self.backend.set_objective(self.objective)

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

    def solve(self, time_limit=None, num_threads=16):
        print "with: " + str(len(self.constraints)) + " constraints."
        print "and " + str(self.g2_vertices_N) + " variables.\n"

        if time_limit != None:
            try:
                self.backend.set_timelimit(time_limit)
            except AttributeError:
                print "WARNING: Unable to set time limit"
                pass

        #self.backend.set_num_threads(num_threads)

        solution = pylp.Solution()
        self.backend.solve(solution)

        return solution
