import csv
import datetime
import math
import os
from json import load
from scipy.stats.stats import spearmanr
from collections import defaultdict
from ..base import Benchmark

METADATA = 'metadata'
BENCHMARK = 'benchmark'
METADATA_EXT = '.json'
PLAINTEXT_EXT = '.txt'
CSV_EXT = '.csv'
OTHER_EXT = 'None'


class Similarity(Benchmark):
    def __init__(self, normalize=True, ignore_oov=True):
        self.normalize = normalize
        self.ignore_oov = ignore_oov
        self.test_set_row_length = 3
        self.should_make_words_lower = False
        self.default_distance = 0
        self.minimum_found_pairs_number = 2
        self.default_result_value = None

    @classmethod
    def read_test_sample(cls, row, is_csv=True):
        if is_csv:
            word_1 = row[0]
            word_2 = row[1]
            similarity = row[2]
        else:
            word_1, word_2, similarity = row.strip().split()
        return (word_1, word_2), float(similarity)

    def read_test_set(self, path):
        test = []
        if path.endswith('.csv'):
            with open(path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                head = True
                for row in reader:
                    if len(row) < self.test_set_row_length:
                        continue
                    if not head:
                        test.append(self.read_test_set(row))
                    head = False
        else:
            with open(path) as f:
                for row in f:
                    test.append(self.read_test_sample(row, is_csv=False))
        return test

    def make_words_lower(self, word_1, word_2):
        if self.should_make_words_lower:
            return word_1.lower(), word_2.lower()
        return word_1, word_2

    def evaluate(self, embeddings, data):
        details = []
        results = []
        found_pairs_number = 0
        for (word_1, word_2), similarity in data:
            word_1, word_2 = self.make_words_lower(word_1, word_2)
            if embeddings.has_word(word_1) and embeddings.has_word(word_2) and not math.isnan(
                    embeddings.get_vector(word_1).dot(embeddings.get_vector(word_2))):
                distance = embeddings.get_vector(word_1).dot(embeddings.get_vector(word_2))
                results.append((distance, similarity))
                found_pairs_number += 1
                details.append([word_1, word_2, float(distance), float(similarity)])
            else:
                if not self.ignore_oov:
                    results.append((self.default_distance, similarity))
                    details.append([word_1, word_2, float(self.default_distance), float(similarity)])
        if len(results) <= self.minimum_found_pairs_number:
            details = []
            return self.default_result_value, found_pairs_number, details
        actual, expected = zip(*results)
        return spearmanr(actual, expected), found_pairs_number, details

    def read_single_dataset(self, path_to_dir, file_name):
        dataset_name, file_extension = os.path.splitext(file_name)
        if file_extension == METADATA_EXT:
            with open(os.path.join(path_to_dir, file_name)) as f:
                data = load(f, strict=False)
            return METADATA, dataset_name, data
        elif file_extension == PLAINTEXT_EXT:
            data = self.read_test_set(os.path.join(path_to_dir, file_name))
            return BENCHMARK, dataset_name, data
        elif file_extension == CSV_EXT:
            data = self.read_test_set(os.path.join(path_to_dir, file_name))
            return BENCHMARK, dataset_name, data
        else:
            return OTHER_EXT, None, None

    def read_datasets_from_dir(self, path_to_dir):
        datasets = defaultdict(lambda: {})
        for file in os.listdir(path_to_dir):
            type, dataset_name, dataset_data = self.read_single_dataset(path_to_dir, file)
            if type != OTHER_EXT:
                datasets[dataset_name][type] = dataset_data
        return datasets

    @classmethod
    def make_metadata_dict(cls, metadata, found_pairs, benchmark_len, dataset_name, embeddings_metadata):
        experiment_setup = dict()
        experiment_setup['cnt_found_pairs_total'] = found_pairs
        experiment_setup['cnt_pairs_total'] = benchmark_len
        experiment_setup['embeddings'] = embeddings_metadata
        experiment_setup['category'] = 'default'
        experiment_setup['dataset'] = dataset_name
        experiment_setup['method'] = 'cosine_distance'
        experiment_setup['language'] = metadata['language']
        experiment_setup['description'] = metadata['description']
        experiment_setup['version'] = metadata['version']
        experiment_setup['measurement'] = 'spearman'
        experiment_setup['task'] = metadata['task']
        experiment_setup['timestamp'] = datetime.datetime.now().isoformat()
        return experiment_setup

    @classmethod
    def make_result(cls, result, details, metadata_dict):
        out = dict()
        out["experiment_setup"] = metadata_dict
        out['correlation'] = result.correlation
        out['p-value'] = result.pvalue
        out['details'] = details
        return out

    def run(self, embs, path_dataset):
        results = []
        datasets = self.read_datasets_from_dir(path_dataset)
        for dataset_name, dataset_data in datasets.items():
            result, cnt_found_pairs_total, details = self.evaluate(embs, dataset_data[BENCHMARK])
            metadata_dict = self.make_metadata_dict(dataset_data[METADATA],
                                                    found_pairs=cnt_found_pairs_total,
                                                    benchmark_len=len(dataset_data[BENCHMARK]),
                                                    dataset_name=dataset_name,
                                                    embeddings_metadata=embs.metadata)
            results.append(self.make_result(result, details, metadata_dict))
        return results

    def get_result(self, embeddings, path_dataset):
        if self.normalize:
            embeddings.normalize()
        results = self.run(embeddings, path_dataset)
        return results
