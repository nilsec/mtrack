import numpy as np
import os


class CoreScheduler(object):
    def __init__(self, cores):
        self.cores = cores

        self.running = set()
        self.finished = set()

    def request_core(self):
        for core in self.cores:
            if not core.id in self.finished:
                if not core.id in self.running:
                    core_nbs = core.nbs
                    if not (self.running & core_nbs):
                        self.running.add(core.id)
                
                        return core

        return None

    def finish_core(self, core_id):
        self.finished.add(core_id)
        self.running.remove(core_id)
