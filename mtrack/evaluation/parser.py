import os
import numpy as np
import json


class EvaluationParser(object):
    def __init__(self, dir_eval):
        # Get pathes:
        self.dir_evaluation = dir_eval

        self.path_chunk_evaluation = os.path.join(self.dir_evaluation,
                                                  "chunk_evaluation.json")

        self.path_line_evaluation = os.path.join(self.dir_evaluation,
                                                 "line_evaluation.json")

        # Load data:
        self.evaluation_chunk = json.load(open(self.path_chunk_evaluation, "r"))
        self.evaluation_line = json.load(open(self.path_line_evaluation, "r"))

        f_beta = lambda tp, fn, fp, beta:\
            ((1 + float(beta)**2) * float(tp))/((1+beta**2) * tp + beta**2 * fn + fp)

        self.f_score_c = {"f_1": f_beta(self.evaluation_chunk["tp"], 
                                        self.evaluation_chunk["fn"], 
                                        self.evaluation_chunk["fp"], 1.0),
                          "f_2": f_beta(self.evaluation_chunk["tp"], 
                                        self.evaluation_chunk["fn"], 
                                        self.evaluation_chunk["fp"], 2.0),
                          "f_05": f_beta(self.evaluation_chunk["tp"], 
                                         self.evaluation_chunk["fn"], 
                                         self.evaluation_chunk["fp"], 0.5)}

        self.f_score_l = {"f_1": f_beta(self.evaluation_line["tp"], 
                                        self.evaluation_line["fn"], 
                                        self.evaluation_line["fp"], 1.0),
                          "f_2": f_beta(self.evaluation_line["tp"], 
                                        self.evaluation_line["fn"], 
                                        self.evaluation_line["fp"], 2.0),
                          "f_05": f_beta(self.evaluation_line["tp"], 
                                         self.evaluation_line["fn"], 
                                         self.evaluation_line["fp"], 0.5)} 
