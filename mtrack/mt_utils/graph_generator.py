import numpy as np
from numpy.random import randint, binomial

from mtrack.graphs import G1
from mtrack.preprocessing import connect_graph_locally


class GraphGenerator(object):
    def __init__(self,
                 n_vertices,
                 min_pos,
                 max_pos,
                 prob_has_partner=0.5):

        self.n_vertices = n_vertices
        self.min_pos = min_pos
        self.max_pos = max_pos

        self.g1 = G1(n_vertices)
        
        print "Generate vertices..."
        for v in self.g1.get_vertex_iterator():
            if (int(v) >=1) and (self.g1.get_partner(int(v) - 1) == int(v)):
                has_partner=True

                self.g1.set_partner(v, int(v) - 1)
                pos_partner = np.array(self.g1.get_position(int(v) - 1))
                ori_partner = np.array(self.g1.get_orientation(int(v) - 1))

                self.g1.set_position(v, pos_partner)
                self.g1.set_orientation(v, -ori_partner)

            else: 
                pos = randint(min_pos, max_pos, 3).astype(float)

                ori = randint(min_pos, max_pos, 3).astype(float)
                ori = ori/np.linalg.norm(ori)

                self.g1.set_position(v, pos)
                self.g1.set_orientation(v, pos)

                if (n_vertices - 1 > int(v) >= 1) and (self.g1.get_partner(int(v)-1) == (-1)):
                    has_partner = bool(binomial(1, prob_has_partner, 1)[0])
                    if has_partner:
                        self.g1.set_partner(v, int(v) + 1)

        self.__validate_vertices()

    
    def __validate_vertices(self):
        for v in self.g1.get_vertex_iterator():
            partner_v = self.g1.get_partner(v)

            if partner_v != (-1):
                pos_v = np.array(self.g1.get_position(v))
                ori_v = np.array(self.g1.get_orientation(v))

                pos_partner = np.array(self.g1.get_position(partner_v))
                ori_partner = np.array(self.g1.get_orientation(partner_v))

                assert(np.all(pos_v == pos_partner))
                assert(np.all(ori_v == -ori_partner))

    def __validate_connectivity(self):
        for e in self.g1.get_edge_iterator():
            p0 = self.g1.get_partner(e.source())
            p1 = self.g1.get_partner(e.target())

            assert(p0 != e.target())
            assert(p1 != e.source())
				
    def random_connectivity(self, n_edges):
        edges_added = []

        while len(edges_added) < n_edges:
            e_random = tuple(np.sort(randint(0, self.n_vertices, 2)))
            
            if e_random[0] != e_random[1]:
                p0 = self.g1.get_partner(e_random[0])
                p1 = self.g1.get_partner(e_random[1])

                if int(p0) == e_random[1]:
                    assert(p1 == e_random[0])
                    continue

                if int(p1) == e_random[0]:
                    assert(p0 == e_random[1])
                    continue

                if not e_random in edges_added:
                    self.g1.add_edge(*e_random)
                    edges_added.append(e_random)

        assert(self.g1.get_number_of_edges() == n_edges)

        self.__validate_connectivity()
        return self.g1

    def local_connectivity(self, distance_threshold):
        self.g1 = connect_graph_locally(self.g1,
                                        distance_threshold)

        self.__validate_connectivity()
        return self.g1
