import datetime
from scipy.stats.stats import spearmanr
import os
import math
from ..base import Benchmark


class Similarity(Benchmark):

    def __init__(self, normalize=True, ignore_oov=True):
        self.normalize = normalize
        self.ignore_oov = ignore_oov

    def read_test_set(self, path):
        test = []
        with open(path) as f:
            for line in f:
                # line = line.lower();
                x, y, sim = line.strip().split()
                test.append(((x, y), float(sim)))
        return test

    def evaluate(self, embs, data):
        details = []
        results = []
        cnt_found_pairs_total = 0
        for (x, y), sim in data:
            x = x.lower()
            y = y.lower()
            # print(x,y)
            if embs.has_word(x) and embs.has_word(y) and not math.isnan(embs.get_vector(x).dot(embs.get_vector(y))):
                # print(m.get_row(x).dot(m.get_row(y)))
                v = embs.get_vector(x).dot(embs.get_vector(y))
                results.append((v, sim))
                cnt_found_pairs_total += 1
                details.append([x, y, float(v), float(sim)])
            else:
                if not self.ignore_oov:
                    # results.append((-1, sim))
                    # details.append([x, y, str(-1), str(sim)])
                    results.append((0, sim))
                    details.append([x, y, str(0), str(sim)])
                    # print('oov')
                    pass
        if len(results) <= 2:
            return -1, cnt_found_pairs_total, []
        actual, expected = zip(*results)
        # print(actual)
        return spearmanr(actual, expected)[0], cnt_found_pairs_total, details

    def run(self, embs, path_dataset):
        results = []
        for file in os.listdir(path_dataset):
            testset = self.read_test_set(os.path.join(path_dataset, file))

            out = dict()
            out["result"], cnt_found_pairs_total, out["details"] = self.evaluate(embs, testset)

            experiment_setup = dict()
            experiment_setup["cnt_found_pairs_total"] = cnt_found_pairs_total
            experiment_setup["cnt_pairs_total"] = len(testset)
            experiment_setup["embeddings"] = embs.metadata
            experiment_setup["category"] = "default"
            experiment_setup["dataset"] = os.path.splitext(file)[0]
            experiment_setup["method"] = "cosine_distance"
            experiment_setup["measurement"] = "spearman"
            experiment_setup["task"] = "word_similarity"
            experiment_setup["timestamp"] = datetime.datetime.now().isoformat()
            out["experiment_setup"] = experiment_setup
            results.append(out)

        return results

    def get_result(self, embs, path_dataset):
        if self.normalize:
            embs.normalize()

        results = self.run(embs, path_dataset)
        return results
