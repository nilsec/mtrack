from mtrack.graphs import GraphConverter, g1_graph
from cgraph_converter import CGraphConverter
import numpy as np

from numpy.random import randint, randn
import time

class RandomInputGraph(object):
    def __init__(self, N_vertices, Nmax_edges):
        self.N = N_vertices
        self.g1 = g1_graph.G1(self.N)

        for i in range(self.N):
            self.g1.set_position(i, randn(3))
            self.g1.set_orientation(i, randn(3))

        for j in range(Nmax_edges):
            edge = randint(0, N_vertices, size=2)

            success = False
            trials = 0
            
            while not success and trials<10: 
                try:
                    self.g1.add_edge(edge[0], edge[1])
                    trials += 1
                    success = True
                except AssertionError:
                    trials += 1
                    pass
                
    def benchmark(self):
        t0 = time.time()
        graph_converter = GraphConverter(self.g1)
        self.g2, self.index_maps = graph_converter.get_g2_graph()
        t1 = time.time()

        print "Converting graph with {} vertices and {} edges took: ".format(self.N, 
                                                                    self.g1.get_number_of_edges())
        print " {} seconds".format(t1 - t0)

    def benchmark_cython(self):
        t0 = time.time()
        graph_converter = CGraphConverter(self.g1)
        self.g2, self.index_maps = graph_converter.get_g2_graph()
        t1 = time.time()

        print "Converting graph with {} vertices and {} edges took: ".format(self.N, 
                                                                    self.g1.get_number_of_edges())
        print " {} seconds".format(t1 - t0)



if __name__ == "__main__":
    rig = RandomInputGraph(1000, 1000)
    rig.benchmark()
    rig.benchmark_cython()

        
        
