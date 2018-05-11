import numpy as np
import os

from mtrack.cores import Core
from gen_context import get_core_cfs


class CoreBuilder(object):
    def __init__(self, 
                 volume_size,
                 core_size,
                 context_size,
                 offset=np.array([0,0,0])):

        """
        volume_size: [x,y,z]
        core/context_size: [x,y,z]
        offset: [x,y,z]
        """

        self.volume_size = np.array(volume_size, dtype=int)
        self.core_size = np.array(core_size, dtype=int)
        self.context_size = np.array(context_size, dtype=int)
        self.offset = np.array(offset, dtype=int)

        self.effective_volume_size = self.volume_size - 2 * self.context_size
        if np.any(self.effective_volume_size <= 0):
            raise Warning("No cores in volume - context too large")
        if not np.all(self.effective_volume_size % self.core_size == np.array([0]*3)):
            raise ValueError("Cores should tile the effective volume without overlap")

        self.block_size = self.core_size + self.context_size
        

    def generate_cores(self, gen_nbs=False):
        """
        TODO: Replace loops
        """
        n_cores = self.effective_volume_size/self.core_size
        max_ids = reduce(lambda x,y: x*y, n_cores)        
        
        cores = []
        core_ids = []
        core_id = 0
        for x_core in range(n_cores[0]):
            x_0 = self.context_size[0] + x_core * self.core_size[0]
 
            for y_core in range(n_cores[1]):
                y_0 = self.context_size[1] + y_core * self.core_size[1]
 
                for z_core in range(n_cores[2]):
                    z_0 = self.context_size[2] + z_core * self.core_size[2]

                    core = Core(x_lim={"min": x_0 + self.offset[0], 
                                       "max": x_0 + self.core_size[0] + self.offset[0]},
                                y_lim={"min": y_0 + self.offset[1], 
                                       "max": y_0 + self.core_size[1] + self.offset[1]},
                                z_lim={"min": z_0 + self.offset[2], 
                                       "max": z_0 + self.core_size[2] + self.offset[2]},
                                context=self.context_size,
                                core_id=core_id,
                                nbs=None)

                    cores.append(core)
                    core_ids.append(core.id)
                    core_id += 1

        if gen_nbs:
            nbs_dict = self._gen_all_nbs(cores)
            for core in cores:
                core.nbs = nbs_dict[core.id]

        assert(np.all(np.arange(len(core_ids)) == np.array(core_ids)))

        return cores

    def gen_cfs(self):
        return get_core_cfs(self.core_size, 
                            self.context_size, 
                            self.volume_size, 
                            pad=True, 
                            debug=True)
         
    def _gen_all_nbs(self, cores):
        """
        LEGACY CODE - Only used for testing.
        """

        print "Generate core neighbours..."
        nbs = {core.id: set() for core in cores}
        
        for i in range(len(cores)):
            for j in range(len(cores)):
                if j > i:
                    i_context = [cores[i].x_lim_context,
                                 cores[i].y_lim_context,
                                 cores[i].z_lim_context]

                    j_context = [cores[j].x_lim_context,
                                 cores[j].y_lim_context,
                                 cores[j].z_lim_context]

                    ovlp = []
                    for dim in range(3):
                        ovlp_dim = set(range(int(i_context[dim]["min"]), int(i_context[dim]["max"]))) &\
                                   set(range(int(j_context[dim]["min"]), int(j_context[dim]["max"])))
                        ovlp.append(bool(ovlp_dim))

                    if np.all(np.array(ovlp)):
                        nbs[cores[i].id].add(cores[j].id)
                        nbs[cores[j].id].add(cores[i].id)

        return nbs
