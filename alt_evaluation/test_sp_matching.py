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
        rec_chains, gt_volume = sp_matching.get_rec_chains([self.line_list[0], self.line_list[1]],
                                                [self.line_list[2], self.line_list[3]],
                                                dimensions=[9,9,12],
                                                tolerance_radius=30.0,
                                                voxel_size=[5.,5.,50.])

        for chain in rec_chains:
            chain.get_shortest_path(plot=False)

class MergeTestCase(unittest.TestCase):
    def setUp(self):
        N = 10
        gt_line_1 = graphs.g1_graph.G1(N)
        gt_line_2 = graphs.g1_graph.G1(N)

        rec_line_1 = graphs.g1_graph.G1(N)

        #Make line:
        for v in range(N - 1):
            gt_line_1.add_edge(v, v+1)
            gt_line_2.add_edge(v, v+1)

            rec_line_1.add_edge(v, v+1)

        z = 2.0
        for v in gt_line_1.get_vertex_iterator():
            gt_line_1.set_position(v, np.array([3.0, 3.0, z]))
            gt_line_2.set_position(v, np.array([9.0, 9.0, z]))
            z += 1

        pos_rec =  [np.array([4.0, 4.0, 2.0]),
                    np.array([4.0, 4.0, 3.0]),
                    np.array([4.0, 4.0, 4.0]),
                    np.array([5.0, 5.0, 5.0]), 
                    np.array([6.0, 6.0, 6.0]), 
                    np.array([7.0, 7.0, 7.0]),
                    np.array([8.0, 8.0, 8.0]),
                    np.array([8.0, 8.0, 9.0]),
                    np.array([8.0, 8.0, 10.0]),
                    np.array([8.0, 8.0, 11.0])]

        for v, pos in zip(rec_line_1.get_vertex_iterator(), pos_rec):
            rec_line_1.set_position(v, pos)

        self.gt_lines = [gt_line_1, gt_line_2]
        self.rec_lines = [rec_line_1]

        self.report = {"fp": 0, "fn": 0, "mergers": 1, "splits": 0}
           
class SplitTestCase(unittest.TestCase):
    def setUp(self):
        N = 10
        gt_line_1 = graphs.g1_graph.G1(N)

        rec_line_1 = graphs.g1_graph.G1(N/2)
        rec_line_2 = graphs.g1_graph.G1(N/2)

        #Make line:
        for v in range(N - 1):
            gt_line_1.add_edge(v, v+1)

        for v in range(N/2 - 1):
            rec_line_1.add_edge(v, v+1)
            rec_line_2.add_edge(v, v+1)

        z = 2.0
        for v in gt_line_1.get_vertex_iterator():
            gt_line_1.set_position(v, np.array([3.0, 3.0, z]))
            z += 1

        z = 2.0
        for v in rec_line_1.get_vertex_iterator():
            rec_line_1.set_position(v, np.array([4.0, 4.0, z]))
            rec_line_2.set_position(v, np.array([2.0, 2.0, z + 5]))
            z += 1

        self.gt_lines = [gt_line_1]
        self.rec_lines = [rec_line_1, rec_line_2]

        self.report = {"fp": 0, "fn": 0, "mergers": 0, "splits": 1}
 


class FnTestCase(unittest.TestCase):
    def setUp(self):
        N = 10
        gt_line_1 = graphs.g1_graph.G1(N)
        gt_line_2 = graphs.g1_graph.G1(N)

        rec_line_1 = graphs.g1_graph.G1(N/2)
        rec_line_2 = graphs.g1_graph.G1(N/2)

        #Make line:
        for v in range(N - 1):
            gt_line_1.add_edge(v, v+1)
            gt_line_2.add_edge(v, v+1)

        for v in range(N/2 - 1):
            rec_line_1.add_edge(v, v+1)
            rec_line_2.add_edge(v, v+1)

        z = 2.0
        for v in gt_line_1.get_vertex_iterator():
            gt_line_1.set_position(v, np.array([3.0, 3.0, z]))
            gt_line_2.set_position(v, np.array([9.0, 9.0, z]))
            z += 1

        z = 2.0
        for v in rec_line_1.get_vertex_iterator():
            rec_line_1.set_position(v, np.array([4.0, 4.0, z]))
            rec_line_2.set_position(v, np.array([2.0, 2.0, z + 5]))
            z += 1

        self.gt_lines = [gt_line_1, gt_line_2]
        self.rec_lines = [rec_line_1, rec_line_2]

        self.report = {"fp": 0, "fn": 0, "mergers": 1, "splits": 1}

class FpTestCase(unittest.TestCase):
    def setUp(self):
        N = 10
        gt_line_1 = graphs.g1_graph.G1(N)

        rec_line_1 = graphs.g1_graph.G1(N/2)

        rec_line_2 = graphs.g1_graph.G1(N)

        #Make line:
        for v in range(N - 1):
            gt_line_1.add_edge(v, v+1)
            rec_line_2.add_edge(v, v+1)

        for v in range(N/2 - 1):
            rec_line_1.add_edge(v, v+1)

        z = 2.0
        for v in gt_line_1.get_vertex_iterator():
            gt_line_1.set_position(v, np.array([3.0, 3.0, z]))
            rec_line_2.set_position(v, np.array([9.0, 9.0, z]))
            z += 1

        z = 2.0
        for v in rec_line_1.get_vertex_iterator():
            rec_line_1.set_position(v, np.array([4.0, 4.0, z]))
            z += 1

        self.gt_lines = [gt_line_1]
        self.rec_lines = [rec_line_1, rec_line_2]

        self.report = {"fp": 1, "fn": 0, "mergers": 0, "splits": 0}


class ErrorGraphTestCase(FpTestCase):
    def runTest(self):
        rec_chains, gt_volume = sp_matching.get_rec_chains(self.rec_lines,
                                                self.gt_lines,
                                                dimensions=[12, 12, 12],
                                                tolerance_radius=20.0,
                                                voxel_size=[5.,5.,50.])

        for chain in rec_chains:
            chain.get_shortest_path(plot=True)

        error_graph = sp_matching.ErrorGraph(rec_chains, gt_volume)
        error_graph.get_sp_matching(3)
        error_graph.draw()
        self.assertEqual(error_graph.get_report(), self.report)

class ShortestPathEvalTestCase(unittest.TestCase):
    def runTest(self):
        gt_line_dir = "/media/nilsec/d0/gt_mt_data/test_tracing/lines_v17_cropped"
        rec_line_dir = "/media/nilsec/d0/gt_mt_data/experiments/selection_cost_grid0404_solve_4/lines"
        dimensions = [1025, 1025, 101]
        tolerance = 50

        print sp_matching.shortest_path_eval(gt_line_dir,
                                       rec_line_dir,
                                       dimensions,
                                       [0,0,300],
                                       tolerance,
                                       [5.,5.,50.],
                                       3)
 
if __name__ == "__main__":
    unittest.main()
