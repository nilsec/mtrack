import numpy as np
import pylp
import os

class G2Solver:
    def __init__(self, g2, backend="Gurobi"):
        self.g2_vertices_N = g2.get_number_of_vertices()
        
        if backend == "Gurobi":
            print "Use Gurobi backend"
            self.backend = pylp.create_linear_solver(pylp.Preference.Gurobi)
        elif backend == "Scip":
            print "Use Scip backend"
            self.backend = pylp.create_linear_solver(pylp.Preference.Scip)
        else:
            raise NotImplementedError("Choose between Gurobi or Scip backend")

        self.backend.initialize(self.g2_vertices_N, pylp.VariableType.Binary)
        self.backend.set_num_threads(1)
        self.objective = pylp.LinearObjective(self.g2_vertices_N)
        
        #pylp.set_log_level(pylp.LogLevel.Debug)

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

        for conflict in g2.get_conflicts():

            constraint = pylp.LinearConstraint()

            all_solved = True
            for v in conflict:
                if not g2.get_solved(v):
                    all_solved=False
                constraint.set_coefficient(v, 1)

            if not all_solved:
                constraint.set_relation(pylp.Relation.LessEqual)
                constraint.set_value(1)
                self.constraints.add(constraint)

        for sum_constraint in g2.get_sum_constraints():

            vertices_1 = sum_constraint[0]
            vertices_2 = sum_constraint[1]

            constraint = pylp.LinearConstraint()

            for v in vertices_1:
                constraint.set_coefficient(v, 1)
            for v in vertices_2:
                constraint.set_coefficient(v, -1)

            all_solved = True
            for v in vertices_1 + vertices_2:
                if not g2.get_solved(v):
                    all_solved = False
                    break
            
            if not all_solved: 
                constraint.set_relation(pylp.Relation.Equal)
                constraint.set_value(0)
                self.constraints.add(constraint)

        for must_pick_one in g2.get_must_pick_one():
            constraint = pylp.LinearConstraint()

            for v in must_pick_one:
                constraint.set_coefficient(v, 1)
                
            constraint.set_relation(pylp.Relation.GreaterEqual)
            constraint.set_value(1)
            self.constraints.add(constraint)

        self.backend.set_constraints(self.constraints)

    def solve(self, time_limit=None, num_threads=16):
        print "with: " + str(len(self.constraints)) + " constraints."
        print "and " + str(self.g2_vertices_N) + " variables.\n"

        if time_limit != None:
            try:
                print "Set time limit of {} seconds".format(time_limit)
                self.backend.set_timeout(time_limit)
            except AttributeError:
                print "WARNING: Unable to set time limit"
                pass

        #self.backend.set_num_threads(num_threads)
        solution, msg = self.backend.solve()
        print "SOLVED with status: " + msg

        return solution
