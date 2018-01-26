class Core(object):
    def __init__(self,
                 x_lim,
                 y_lim,
                 z_lim,
                 context,
                 core_id,
                 nbs):
        
        self.x_lim_core = x_lim
        self.y_lim_core = y_lim
        self.z_lim_core = z_lim

        self.x_lim_context = {"min": x_lim["min"] - context[0], 
                              "max": x_lim["max"] + context[0]}

        self.y_lim_context = {"min": y_lim["min"] - context[1], 
                              "max": y_lim["max"] + context[1]}
 
        self.z_lim_context = {"min": z_lim["min"] - context[2], 
                              "max": z_lim["max"] + context[2]}
        
        self.id = core_id
        self.nbs = nbs
