import unittest
import numpy as np

from mtrack.cores import get_cfs, get_core_cfs
from mtrack.cores import CoreBuilder
from core_builder import CoreBaseTest


class GenContextBaseTest(unittest.TestCase):
    def setUp(self):
        self.volume_size = np.array([60, 60, 60])
        self.core_size = np.array([10,10,10])
        self.context_size = np.array([10,10,10])
        self.offset = np.array([0,0,0])
        

class GenContextTestCase(GenContextBaseTest):
    def runTest(self):

        for cs in [[0,0,0], 
                   [10,10,10], 
                   [15,15,15], 
                   [5,10,15],
                   [15,10,5],
                   [10,5,15]]:

            self.context_size = np.array(cs)

            if cs[0] != cs[1]:
                degenerate = True
            else:
                degenerate = False

            builder = CoreBuilder(volume_size=self.volume_size,
                                  core_size=self.core_size,
                                  offset=self.offset,
                                  context_size=self.context_size)

            cores = builder.generate_cores(gen_nbs=True)

            # Completeness of generated cores checked with debug=True
            cf_lists = get_core_cfs(self.core_size, 
                                    self.context_size, 
                                    self.volume_size,
                                    pad=True,
                                    debug=True)

            # Total number of cores has to be equal:
            self.assertTrue(len(cores) == len([j for k in cf_lists for j in k]))

            # Check against naive nb generator that no constraints are violated:
            for cf_list in cf_lists:
                for cf_id in cf_list:
                    for core in cores:
                        if core.id == cf_id:
                            self.assertFalse(set(core.nbs) & set(cf_list))

            if not degenerate:
                # Check that we found the largest cf lists if non-degenerate inputs:
                cf_lists_cores = []
                done = []
                for core in cores:
                    cf_list_core = []
                    if not core.id in done:
                        cf_list_core.append(core)
                        nbs_all = list(core.nbs)
                        for core_2 in cores:
                            if not core_2.id == core.id:
                                if not core_2.id in nbs_all:
                                    nbs_all += list(core_2.nbs)
                                    cf_list_core.append(core_2.id)

                    done.append(core.id)
                    cf_lists_cores.append(len(cf_list_core))

                 
                    l = [len(k) for k in cf_lists]
                    self.assertEqual(max(l), max(cf_lists_cores))


if __name__ == "__main__":
    unittest.main()
