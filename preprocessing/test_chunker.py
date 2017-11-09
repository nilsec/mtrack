from chunker import Chunker
import numpy as np
import unittest

class TestDivideChunk(unittest.TestCase):
    def runTest(self):
        volume_shape = np.array([1000, 1024, 1024])
        max_chunk_shape = np.array([75, 1024, 1024])
        voxel_size = [5.,5.,50.]
        overlap=np.array([20, 20, 5])
        
        chunker = Chunker(volume_shape,
                          max_chunk_shape,
                          voxel_size,
                          overlap)

        print chunker.volume.shape
        chunks = chunker.chunk()
        for c in chunks:
            print c.limits
            print c.shape

if __name__ == "__main__":
    unittest.main()

