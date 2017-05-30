from graph import G
import numpy as np

class G2(G):
    def __init__(self, N):
        G.__init__(self, N)

        # Initialized with 0.0, see graph.G and tests
        G.new_vertex_property(self, "costs", dtype="double")

        G.new_graph_property(self, "conflicts", dtype="python::object")
        G.set_graph_property(self, "conflicts", set())

        G.new_graph_property(self, "sum_constraints", dtype="python::object")
        G.set_graph_property(self, "sum_constraints", list())

    def __check_id(self, u):
        assert(type(u) == int)
        assert(u < G.get_number_of_vertices(self))
        assert(u >= 0)

    def add_conflict(self, exclusive_vertices):
        """
        Add a tuple of exclusive vertices.
        """
        for u in exclusive_vertices:
            self.__check_id(u)

        conflicts = G.get_graph_property(self, "conflicts")
        conflicts.add(tuple(exclusive_vertices))
        G.set_graph_property(self, "conflicts", conflicts)

        return conflicts

    def get_conflicts(self):
        return G.get_graph_property(self, "conflicts")

    def add_sum_constraint(self, vertices_left, vertices_right):
        """
        Require that the number of selected G2 nodes centered
        around a G1 node u is equal to the left and right
        of u. Ensures that we do not get branching.
        """
        assert(type(vertices_left) == list)
        assert(type(vertices_right) == list)

        for u in vertices_left + vertices_right:
            self.__check_id(u)
        
        sum_constraints = G.get_graph_property(self, "sum_constraints")
        sum_constraints.append(tuple([vertices_left, vertices_right]))
        G.set_graph_property(self, "sum_constraints", sum_constraints)

    def get_sum_constraints(self):
        return G.get_graph_property(self, "sum_constraints")

    def set_cost(self, u, value):
        G.set_vertex_property(self, "costs", u, value)

    def get_cost(self, u):
        return G.get_vertex_property(self, "costs", u)


        
