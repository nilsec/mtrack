import os
import numpy as np
import json


class EvaluationParser(object):
    def __init__(self, node_stats_path):
        self.node_stats_path = node_stats_path

        # Load data:
        self.node_stats = json.load(open(self.node_stats_path, "r"))


        # Splits are defined on rec tracks. Whenever two gt trajectories
        # are matched to one gt we split one gt track. Similarly 
        # False positives are defined on rec tracks thus we define:
        # FP = False Positives + Splits
        # Similarly: FN = False Negatives + Merges

        self.tps = self.node_stats["tps_gt"]
        self.fps = self.node_stats["fps"]
        self.fns = self.node_stats["fns"]
        self.merges = self.node_stats["merges"]
        self.splits = self.node_stats["splits"]

        self.precision = float(self.tps)/(self.fps + self.tps)
        self.recall = float(self.tps)/(self.fns + self.tps)

        self.split_precision = float(self.tps)/(self.fps + self.splits + self.tps)
        self.merge_recall = float(self.tps)/(self.fns + self.merges + self.tps)

    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall

    def get_split_precision(self):
        return self.split_precision

    def get_merge_recall(self):
        return self.merge_recall
