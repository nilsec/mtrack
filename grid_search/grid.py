import solve
from collections import deque
import itertools
import json
import os
from functools import partial
import pickle
import pprint
import sys
import time
from redirect_output import RedirectOutput
from preprocessing import DirectionType

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DirectionType):
            return {"perp": obj.perp, "par": obj.par}
        return json.JSONEncoder.default(self, obj)

class Grid:
    def __init__(self, parameter=None, grid_object=None):
        if grid_object is None:
        
            if parameter is None:
                self.parameter = {}        
            else:
                self.parameter = parameter

            self.archive = {}
            self._grid = None
            self.n_run = 0
            self.n_tot = 0
            self.from_grid_object = False

        else:
            if isinstance(grid_object, str):
                if os.path.exists(grid_object):
                    self.load(grid_object)
                else:
                    raise FileNotFoundError("Grid object does not exist at specified path.")
            
            elif isinstance(grid_object, Grid):
                self.__dict__.update(grid_object.__dict__)

            else:
                raise TypeError("Input grid format not understood.")

            self.from_grid_object = True
            print self._grid

            if not self._grid:
                raise Warning("Grid empty/finished")
 
    def add_parameter(self, param_name, param_values):
        param_values = list(param_values)
        
        try:
            print "Parameter already present. Extend to: \n"
            self.parameter[param_name].extend(param_values)
            print self.parameter[param_name], "\n"

        except TypeError:
            self.parameter[param_name] = param_values


    def __getattr__(self, param_name):
        try:
            return self.parameter[str(param_name)]
        except KeyError:
            raise AttributeError("Parameter not defined")


    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self.__dict__, f, protocol=2)
        f.close()


    def load(self, path):
        f = open(path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
        
        if not self._grid:
            print "Grid empty/finished"

        self.from_grid_object = True
        

    def run(self, f_solve, f_solve_parameter, 
            f_eval=None, f_eval_parameter=None,
            verbose=True, save_grid=True, skip_runs=[]):
        
        if f_eval is not None:
            if f_eval_parameter is not None:
                assert(isinstance(f_eval_parameter, dict))
                f_eval_partial = partial(f_eval, **f_eval_parameter)

        assert(isinstance(f_solve_parameter, dict))
        
        try:
            grid_base_dir = f_solve_parameter["output_dir"]
            del f_solve_parameter["output_dir"]

        except KeyError:
            raise TypeError("f_solve needs to have an \"output_directory\" keyword " + \
                             "corresponding to the Grid base directory.")

        # Ugly, replace later
        count_params = {}
        for key, value in f_solve_parameter.iteritems():
            if isinstance(value, str):
                if "n_run" in value:
                    count_params[key] = value

        for key in count_params:
            del f_solve_parameter[key]
 
        
        if not self.from_grid_object:
            self._grid = deque(dict(zip(self.parameter, x)) 
                               for x in itertools.product(*self.parameter.itervalues()))

            self.n_tot = len(self._grid)
        
        print "Start Grid Search...\n"
        f_solve_partial = partial(f_solve, **f_solve_parameter)

        while self._grid:
            self.n_run += 1
            if self.n_run in skip_runs:
                self._grid.pop()
                print "Skip run number %s" % self.n_run
                continue
 
            print "Run " + str(self.n_run) + "/" + str(self.n_tot) + \
                  "\n----------------------------------------\n"
          
            # Ugly, replace later 
            count_params_tmp = {}
            for key, value in count_params.iteritems():
                count_params_tmp[key] = value.replace("n_run", str(self.n_run))
  
            f_solve_partial = partial(f_solve_partial, **count_params_tmp)
            output_directory = grid_base_dir + "/grid_%s/" % self.n_run
            
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            p_run = self._grid.pop()
            p_run["output_dir"] = output_directory
            
            if verbose:
                pp = pprint.PrettyPrinter(indent=4)
                print "Solving for:"
                pp.pprint(p_run), "\n\n"
            
            
            with RedirectOutput(stdout=output_directory + "f_solve.log",
                                stderr=output_directory + "f_solve.err"):
 
                start_time = time.time()
                solution = f_solve_partial(**p_run)
                solve_time = time.time() - start_time

            self.archive[self.n_run] = {"p_run": p_run, 
                                        "solution": solution, 
                                        "solve_time": solve_time}

            with open(output_directory + "parameter.json", "w+") as f:
                json.dump(p_run, f, sort_keys=True, indent=4, cls=CustomEncoder)
            
            if "f_eval_partial" in locals():
                evaluation = f_eval_partial(solution)
                self.archive[self.n_run]["evaluation"] = evaluation
                with open(output_directory + "evaluation.json", "w+") as f:
                    json.dump(evaluation, f, sort_keys=True, indent=4)

            if save_grid:
                if (self.n_run % 2 == 0):
                    self.save(grid_base_dir + "/grid_0.p")
                else:
                    self.save(grid_base_dir + "/grid_1.p")
            
            print "----------------------------------------\n\n"

