import unittest 
import numpy as np
import graphs
import sp_matching

class SimpleLines(unittest.TestCase):
    def setUp(self):
        N = 5
        g1_line_1 = graphs.g1_graph.G1(N)
        g1_line_2 = graphs.g1_graph.G1(N)
        g1_line_3 = graphs.g1_graph.G1(N)
        g1_line_4 = graphs.g1_graph.G1(N)        

        #Make line:
        for v in range(N - 1):
            g1_line_1.add_edge(v, v+1)
            g1_line_2.add_edge(v, v+1)
            g1_line_3.add_edge(v, v+1)
            g1_line_4.add_edge(v, v+1)
           
 
        z = 3.0
        for v in g1_line_1.get_vertex_iterator():
            g1_line_1.set_position(v, np.array([4.0, 4.0, z]))
            g1_line_2.set_position(v, np.array([0.0, z, z])) 
            g1_line_3.set_position(v, np.array([1.0, z+1, z+1]))
            g1_line_4.set_position(v, np.array([0.0, 0.0, z]))
            z += 1

        self.line_list = [g1_line_1, g1_line_2, g1_line_3, g1_line_4]

class GtVolumeTestCase(SimpleLines):
    def runTest(self):
        volume = sp_matching.get_gt_volume(self.line_list, [9, 9, 12], tolerance_radius=10.0, voxel_size = [5.,5.,50.])
        volume_drawn = volume.get_binary_volume(2, [9, 9, 12])
        #print volume_drawn

class RecChainTestCase(unittest.TestCase):
    def runTest(self):
        rec_chain = sp_matching.RecChain() 
        rec_chain.add_voxel([1])
        rec_chain.add_voxel([1,2,3])
        rec_chain.add_voxel([1,2])
        rec_chain.add_voxel([1,2])
        rec_chain.add_voxel([3,4,2])

        #rec_chain.draw()

class GetRecChainsTestCase(SimpleLines):
    def runTest(self):
        rec_chains = sp_matching.get_rec_chains([self.line_list[0], self.line_list[1]],
                                                [self.line_list[2], self.line_list[3]],
                                                dimensions=[9,9,12],
                                                tolerance_radius=30.0,
                                                voxel_size=[5.,5.,50.])

        for chain in rec_chains:
            chain.get_shortest_path(plot=True)


        

if __name__ == "__main__":
    unittest.main()
        
        
        
