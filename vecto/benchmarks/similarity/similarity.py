import csv
import datetime
import math
import os
from json import load
from scipy.stats.stats import spearmanr
from collections import defaultdict
from ..base import Benchmark

TYPE_METADATA = 'metadata'
TYPE_BENCHMARK = 'benchmark'
METADATA_EXT = '.json'
PLAINTEXT_EXT = '.txt'
CSV_EXT = '.csv'
OTHER_EXT = 'None'


class Similarity(Benchmark):
    def __init__(self, normalize=True, ignore_oov=True):
        self.normalize = normalize
        self.ignore_oov = ignore_oov

    def read_test_set(self, path):
        test = []
        if path.endswith(".csv"):
            with open(path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                head = True
                for row in reader:
                    if len(row) < 3:
                        continue
                    if not head:
                        x = row[0]
                        y = row[1]
                        sim = row[2]
                        test.append(((x, y), float(sim)))
                    head = False
        else:
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

    def read_single_dataset(self, path_to_dir, file_name):
        # TODO: fix this mess
        name_category, file_extension = os.path.splitext(file_name)
        if file_extension == METADATA_EXT:
            with open(os.path.join(path_to_dir, file_name)) as f:
                data = load(f, strict=False)
            return TYPE_METADATA, name_category, data
        elif file_extension == PLAINTEXT_EXT:
            # TODO: use logger
            print(file_name)
            print("loading as plain text")
            data = self.read_test_set(os.path.join(path_to_dir, file_name))
            return TYPE_BENCHMARK, name_category, data
        elif file_extension == CSV_EXT:
            data = self.read_test_set(os.path.join(path_to_dir, file_name))
            return TYPE_BENCHMARK, name_category, data
        else:
            return OTHER_EXT, None, None

    def read_datasets_from_dir(self, path_to_dir):
        datasets = defaultdict(lambda: {})
        for file in os.listdir(path_to_dir):
            type, name_category, dataset_data = self.read_single_dataset(path_to_dir, file)
            if type != OTHER_EXT:
                datasets[name_category][type] = dataset_data
        return datasets

    def make_metadata_dict(self, metadata, found_pairs, benchmark_len, dataset_name, embeddings_metadata):
        experiment_setup = {}
        experiment_setup['cnt_found_pairs_total'] = found_pairs
        experiment_setup['cnt_pairs_total'] = benchmark_len
        experiment_setup['embeddings'] = embeddings_metadata
        experiment_setup['category'] = "default"
        experiment_setup['dataset'] = dataset_name
        experiment_setup['method'] = "cosine_distance"
        # TODO: fix this
        # experiment_setup['language'] = metadata['language']
        # experiment_setup['description'] = metadata['description']
        # experiment_setup['version'] = metadata['version']
        experiment_setup['measurement'] = "spearman"
        experiment_setup['task'] = "word similarity"
        experiment_setup['timestamp'] = datetime.datetime.now().isoformat()
        return experiment_setup

    def make_result(self, result, details, metadata_dict):
        out = {}
        out["experiment_setup"] = metadata_dict
        out["experiment_setup"]["default_measurement"] = "spearman"
        out['result'] = {"spearman": result}
        out['details'] = details
        return out

    def run(self, embeddings, dataset):
        if self.normalize:
            embeddings.normalize()
        results = []
        path_dataset = dataset.path
        datasets = self.read_datasets_from_dir(path_dataset)
        for dataset_name, dataset_data in datasets.items():
            result, cnt_found_pairs_total, details = self.evaluate(embeddings, dataset_data[TYPE_BENCHMARK])
            if TYPE_METADATA not in dataset_data:
                dataset_data[TYPE_METADATA]={}
            metadata_dict = self.make_metadata_dict(dataset_data[TYPE_METADATA],
                                                    found_pairs=cnt_found_pairs_total,
                                                    benchmark_len=len(dataset_data[TYPE_BENCHMARK]),
                                                    dataset_name=dataset_name,
                                                    embeddings_metadata=embeddings.metadata)
            results.append(self.make_result(result, details, metadata_dict))
        return results

