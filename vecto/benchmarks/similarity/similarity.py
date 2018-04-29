import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import sys
import datetime
import yaml
import argparse
from scipy.stats.stats import spearmanr
import os
import random
import math
from ..base import Benchmark

def read_test_set(path):
    test = []
    with open(path) as f:
        for line in f:
            # line = line.lower();
            x, y, sim = line.strip().split()
            test.append(((x, y), float(sim)))
    return test

def evaluate(m, data):
    details = []
    results = []
    count = 0
    for (x, y), sim in data:
        x = x.lower()
        y = y.lower()
        # print(x,y)
        if m.has_word(x) and m.has_word(y) and not math.isnan(m.get_row(x).dot(m.get_row(y))):
            # print(m.get_row(x).dot(m.get_row(y)))
            v =  m.get_row(x).dot(m.get_row(y))
            results.append((v, sim))
            count += 1
            details.append([x, y, float(v), float(sim)])
        else:
            # results.append((-1, sim))
            # details.append([x, y, str(-1), str(sim)])
            results.append((0, sim))
            details.append([x, y, str(0), str(sim)])
            # print('oov')
            pass
    if len(results) <= 2:
        return -1, count, []
    actual, expected = zip(*results)
    # print(actual)
    return spearmanr(actual, expected)[0], count, details


def run(embeddings, datasset_path):
    results = []
    for file in os.listdir(datasset_path):
        testset = read_test_set(os.path.join(datasset_path, file))

        out = dict()
        out["result"], count, out["details"] = evaluate(embeddings, testset)

        experiment_setup = dict()
        experiment_setup["cnt_finded_pairs_total"] = count
        experiment_setup["cnt_pairs_total"] = len(testset)
        experiment_setup["embeddings"] = embeddings.metadata
        experiment_setup["category"] = "default"
        experiment_setup["dataset"] = os.path.splitext(file)[0]
        experiment_setup["method"] = "cosine_distance"
        experiment_setup["measurement"] = "spearman"
        experiment_setup["task"] = "word_similarity"
        experiment_setup["timestamp"] = datetime.datetime.now().isoformat()
        out["experiment_setup"] = experiment_setup
        results.append(out)


    return results


class Similarity(Benchmark):

    def __init__(self):
        self.normalize = True


    def get_result(self, embs, dataset_path):
        if self.normalize:
            embs.normalize()

        results = run(embs, dataset_path)
        return results

