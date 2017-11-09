from chunker import Chunker
from create_probability_map import slices_to_chunks
import numpy as np
import unittest

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

if __name__ == "__main__":
    unittest.main()

