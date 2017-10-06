import os
import numpy as np
import json


class EvaluationParser(object):
    def __init__(self, dir_solution, evaluation_number):
        # Get pathes from solution dir:
        self.dir_evaluation = os.path.join(dir_solution,
                                           "evaluation_{}".format(evaluation_number))

        self.path_chunk_evaluation = os.path.join(self.dir_evaluation,
                                                  "chunk_evaluation.json")

        self.path_line_evaluation = os.path.join(self.dir_evaluation,
                                                 "line_evaluation.json")

        self.path_matching_params = os.path.join(self.dir_evaluation,
                                                 "matching_params.json")

        self.path_solve_params = os.path.join(dir_solution,
                                              "solve_params.json")

        self.path_solve_stats = os.path.join(dir_solution,
                                             "solve_stats.json")

        # Load data:
        self.evaluation_chunk = json.load(open(self.path_chunk_evaluation, "r"))
        self.evaluation_line = json.load(open(self.path_line_evaluation, "r"))
        self.params_matching = json.load(open(self.path_matching_params, "r"))
        self.params_solve = json.load(open(self.path_solve_params, "r"))
        self.stats_solve = json.load(open(self.path_solve_stats, "r"))        
