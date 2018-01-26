import numpy as np
import os

from mtrack.cores import Core


class CoreBuilder(object):
    def __init__(self, 
                 volume_size,
                 core_size,
                 context_size,
                 min_core_overlap=np.array([0,0,0]),
                 offset=np.array([0,0,0])):

        self.volume_size = np.array(volume_size, dtype=float)
        self.core_size = np.array(core_size, dtype=float)
        self.context_size = np.array(context_size, dtype=float)
        self.min_core_overlap = np.array(min_core_overlap, dtype=float)

        self.offset = offset

        self.block_size = self.core_size + self.context_size

        self.cores = []
        self.running = []

    def _gen_all_nbs(self, cores):
        print "Generate core neighbours..."
        nbs = {core.id: set() for core in cores}
        
        
        for i in range(len(cores)):
            print "{}/{}".format(i, len(cores))
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
        
    def _gen_nbs(self, core_id, n_cores):
        """
        Generate all 25 nbs of a core.
        Seemed easy, edge cases make this function
        a beast. In hindsight the edge case exclusions
        probably define the neighbours sufficiently
        and f_core is obsolete. 
        I.e. neighbours are those ids 
        that differ in zy plane for and are not more than
        mod 1 distant in the +- cases. 
        For the 0 case same with mod but stay in same plane.
        """
        n_Z = n_cores[2]
        n_Y = n_cores[1]
        n_X = n_cores[0]

        # Corresponds to x=0 plane
        f_core_0 = lambda j,k: core_id + j + n_Z * k
        
        # x - 1 plane
        f_core_m1 = lambda j,k: core_id + j - ((n_Y + k) * n_Z)

        # x + 1 plane
        f_core_p1 = lambda j,k: core_id + j + ((n_Y + k) * n_Z)

        core_id_range = range(reduce(lambda x,y: x * y, n_cores))

        nbs = set()
        for j in [0,1,-1]:
            for k in [0,1,-1]:
                c0 = f_core_0(j,k)
                cm1 = f_core_m1(j,k)
                cp1 = f_core_p1(j,k)

                if c0 in core_id_range:
                    # Generally the id needs to be in range (1,1,1)
                    if np.all(np.array([abs(c0%n_Z - core_id%n_Z), 
                                        abs(c0%n_Y - core_id%n_Y)]) <= np.array([1,1])):
                
                        if abs(c0/(n_Y * n_Z) - core_id/(n_Y * n_Z)) == 0:
  
                            # In case of +- 1 only z plane change
                            if abs(c0 - core_id) > 1:
                                # id needs to change y+-1 plane
                                if abs(c0/n_Z - core_id/n_Z) == 1:
                                    nbs.add(c0)
                            else:
                                nbs.add(c0)
               
                if np.all(np.array([abs(cm1%n_Z - core_id%n_Z), 
                                    abs(cm1%n_Y - core_id%n_Y)]) <= np.array([1,1])):
                    if cm1 in core_id_range:
                        # id needs to change x-1 plane
                        if abs(cm1/(n_Y * n_Z) - core_id/(n_Y * n_Z)) == 1:
                            nbs.add(cm1)


                if np.all(np.array([abs(cp1%n_Z - core_id%n_Z), 
                                    abs(cp1%n_Y - core_id%n_Y)]) <= np.array([1,1])):
                    if cp1 in core_id_range:
                        # id needs to change to x+1 plane
                        if abs(cp1/(n_Y * n_Z) - core_id/(n_Y * n_Z)) == 1:
                            nbs.add(cp1)
 
        nbs.remove(core_id)

        return nbs

    def generate_cores(self):
        print "Generate cores..."
        if np.all(self.volume_size == self.core_size):
            cores = [Core(x_lim={"min": self.offset[0], 
                               "max": self.core_size[0] + self.offset[0]},
                        y_lim={"min": self.offset[1], 
                               "max": self.core_size[1] + self.offset[1]},
                        z_lim={"min": self.offset[2], 
                               "max": self.core_size[2] + self.offset[2]},
                        context=[0.0,0.0,0.0],
                        core_id=0,
                        nbs=set([]))]
            return cores
            

        n_cores, ovlp = self._get_overlap()
        max_ids = reduce(lambda x,y: x*y, n_cores)        

        x_0 = self.context_size[0]
        y_0 = self.context_size[1]
        z_0 = self.context_size[2]


        cores = []
        core_id = 0
        for x_core in range(n_cores[0]):
            x_0 = self.context_size[0] + x_core * self.core_size[0] - x_core * ovlp[0]
 
            for y_core in range(n_cores[1]):
                y_0 = self.context_size[1] + y_core * self.core_size[1] - y_core * ovlp[1]
 
                for z_core in range(n_cores[2]):
                    z_0 = self.context_size[2] + z_core * self.core_size[2] - z_core * ovlp[2]

                    #nbs = self._gen_nbs(core_id, n_cores)
                                            

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
                    core_id += 1

        nbs_dict = self._gen_all_nbs(cores)
        for core in cores:
            core.nbs = nbs_dict[core.id]

        return cores


    def _get_overlap(self):
        """
        Get actual overlap such that the whole volume is packed with
        cubes of size core_size + context size with a minumum
        core overlap of self.min_core_overlap. The calculated
        overlap will be >= self.min_core_overlap.
        """
        core_volume = self.volume_size - 2*self.context_size
        n_cores = np.ceil(core_volume/self.core_size)
        assert(np.all(n_cores > 1))
        core_volume_novlp = n_cores * self.core_size
        diff = core_volume_novlp - core_volume
        ovlp = diff/(n_cores - 1.0)
 
        while np.any(ovlp < self.min_core_overlap):
            core_volume_novlp = n_cores * self.core_size
            diff = core_volume_novlp - core_volume
            ovlp = diff/(n_cores - 1)
            n_cores[np.where(ovlp < self.min_core_overlap)] += 1
        
        for i in range(3):
            for j in range(3):
                balance = ovlp[i]/ovlp[j]
                if balance < 0.7:
                    print "WARNING, overlap not balanced!"

        return n_cores.astype(int), ovlp
