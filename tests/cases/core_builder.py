import unittest
import numpy as np

from mtrack.cores import CoreBuilder

class CoreBaseTest(unittest.TestCase):
    def setUp(self):
        self.volume_size = np.array([200, 100, 60])
        self.core_size = np.array([10, 10, 10])
        self.offset = np.array([0, 0, 0])
        self.context_size = np.array([20, 20, 20])
        self.n_cores = np.array([16, 6, 2])
        
class GenerateCoresTestCase(CoreBaseTest):
    def runTest(self):
        builder = CoreBuilder(volume_size=self.volume_size,
                              core_size=self.core_size,
                              offset=self.offset,
                              context_size=self.context_size)

        cores = builder.generate_cores(gen_nbs=True)
        # Check that we have the correct number of cores
        self.assertTrue(len(cores) == reduce(lambda x,y: x*y, self.n_cores))

        for core in cores:
            # Check correct core size
            self.assertTrue(core.x_lim_core["max"] - core.x_lim_core["min"] == self.core_size[0])
            self.assertTrue(core.y_lim_core["max"] - core.y_lim_core["min"] == self.core_size[1])
            self.assertTrue(core.z_lim_core["max"] - core.z_lim_core["min"] == self.core_size[2])

            # Check correct context size
            self.assertTrue(core.x_lim_context["max"] -\
                            core.x_lim_context["min"] == 2*self.context_size[0] + self.core_size[0])
            self.assertTrue(core.y_lim_context["max"] -\
                            core.y_lim_context["min"] == 2*self.context_size[1] + self.core_size[1])
            self.assertTrue(core.z_lim_context["max"] -\
                            core.z_lim_context["min"] == 2*self.context_size[2] + self.core_size[2])

            # Check that each block is contained in volume
            self.assertTrue(core.x_lim_context["min"] >= self.offset[0]) 
            self.assertTrue(core.y_lim_context["min"] >= self.offset[1])
            self.assertTrue(core.z_lim_context["min"] >= self.offset[2])

            self.assertTrue(core.x_lim_context["max"] <= self.offset[0] + self.volume_size[0])
            self.assertTrue(core.y_lim_context["max"] <= self.offset[1] + self.volume_size[1])
            self.assertTrue(core.z_lim_context["max"] <= self.offset[2] + self.volume_size[2]) 

            self.assertTrue(core.x_lim_core["min"] >= self.offset[0]) 
            self.assertTrue(core.y_lim_core["min"] >= self.offset[1])
            self.assertTrue(core.z_lim_core["min"] >= self.offset[2])

            self.assertTrue(core.x_lim_core["max"] <= self.offset[0] + self.volume_size[0])
            self.assertTrue(core.y_lim_core["max"] <= self.offset[1] + self.volume_size[1])
            self.assertTrue(core.z_lim_core["max"] <= self.offset[2] + self.volume_size[2]) 

class NeighbourTestCase(CoreBaseTest):
    def runTest(self):
        i = 0
        max_nbs = [0, 26, 124]
        min_nbs = [0, 7, 26]
       
        # Overwrite volume size s.t. max number of neighbours is realized 
        self.volume_size = np.array([200, 160, 120])
    
        for context_size in [np.array([0,0,0]), 
                             np.array([5,5,5]), 
                             np.array([10,10,10])]:

            builder = CoreBuilder(volume_size=self.volume_size,
                                  core_size=self.core_size,
                                  offset=self.offset,
                                  context_size=context_size)

            cores = builder.generate_cores(gen_nbs=True)

            len_nbs = [len(x.nbs) for x in cores]
            self.assertTrue(max(len_nbs) == max_nbs[i])
            self.assertTrue(min(len_nbs) == min_nbs[i])
            i += 1

if __name__ == "__main__":
    unittest.main()
