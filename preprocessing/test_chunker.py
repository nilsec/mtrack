from chunker import Chunker
from create_probability_map import slices_to_chunks
import numpy as np
import unittest
import graphs
"""
class TestDivideChunk(unittest.TestCase):
    def runTest(self):
        volume_shape = np.array([400, 1024, 1024])
        max_chunk_shape = np.array([75, 1024, 1024])
        voxel_size = [5.,5.,50.]
        overlap=np.array([20, 20, 10])
        
        chunker = Chunker(volume_shape,
                          max_chunk_shape,
                          voxel_size,
                          overlap)

        chunks = chunker.chunk()
        slices_to_chunks("/media/nilsec/m1/gt_mt_data/probability_maps/test/parallel",
                         "/media/nilsec/m1/gt_mt_data/probability_maps/test/parallel/chunks",
                         chunks)
"""
class TestIds(unittest.TestCase):
    def runTest(self):
        g1 = graphs.G1(0)
        g1.load("/media/nilsec/m1/gt_mt_data/solve_volumes/chunk_test/solution0/volume.gt")
    
        ids = []
        for v in g1.get_vertex_iterator():
            ids.append(v)

        print len(ids)
        print len(set(ids))
            

if __name__ == "__main__":
    unittest.main()

