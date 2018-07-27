from ..base import Benchmark
from collections import defaultdict
from os import path, listdir
import csv
import numpy as np
from scipy.spatial import distance

OTHER_EXT = 'None'
BENCHMARK = 'benchmark'


class SynonymyDetection(Benchmark):
    def __init__(self, normalize=True,
                 ignore_oov=True,
                 do_top5=True,
                 need_subsample=False,
                 size_cv_test=1,
                 set_aprimes_test=None,
                 inverse_regularization_strength=1.0,
                 exclude=True):
        self.normalize = normalize
        self.ignore_oov = ignore_oov
        self.do_top5 = do_top5
        self.need_subsample = need_subsample
        self.normalize = normalize
        self.size_cv_test = size_cv_test
        self.set_aprimes_test = set_aprimes_test
        self.inverse_regularization_strength = inverse_regularization_strength
        self.exclude = exclude

        self.stats = {}
        self.cnt_total_correct = 0
        self.cnt_total_total = 0

        # this are some hard-coded bits which will be implemented later
        self.result_miss = {
            'rank': -1,
            'reason': 'missing words'
        }

    @property
    def method(self):
        return type(self).__name__

    @classmethod
    def read_test_set(self, path):
        data = defaultdict(lambda: [])
        if path.endswith('.csv'):
            with open(path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                head = True
                for row in reader:
                    if len(row) < 3:
                        continue
                    if not head:
                        target_word = row[1]
                        word = row[2]
                        is_synonym = row[3]
                        data[target_word].append([word, is_synonym])
                    head = False
        else:
            with open(path) as f:
                for line in f:
                    _, target_word, word, is_synonym = line.strip().split()
                    data[target_word].append([word, is_synonym])
        return dict(data)

    def collect_stats(self, embs, data):
        corrected_data = defaultdict(lambda: [])
        for word, suspicious_words in data.items():
            if not embs.has_word(word):
                continue
            for susp_word, is_synonym in suspicious_words:
                if embs.has_word(susp_word):
                    corrected_data[word].append([susp_word, is_synonym])
        result = self.run_synonym_finding(embs, dict(corrected_data))
        return result

    def evaluate(self, embs, data):
        result = self.collect_stats(embs, data)
        return result

    def read_datasets_from_dir(self, path_to_dir):
        datasets = defaultdict(lambda: {})
        for file in listdir(path_to_dir):
            dataset_name, dataset_data = self.read_single_dataset(path_to_dir, file)
            if type != OTHER_EXT:
                datasets[dataset_name] = dataset_data
        return datasets

    def read_single_dataset(self, path_to_dir, file_name):
        dataset_name, _ = path.splitext(file_name)
        data = self.read_test_set(path.join(path_to_dir, file_name))
        return dataset_name, data

    def run(self, embeds, path_dataset):
        results = defaultdict(lambda: {})
        datasets = self.read_datasets_from_dir(path_dataset)
        for dataset_name, dataset_data in datasets.items():
            result = self.evaluate(embeds, dataset_data)
            results[dataset_name] = result
        return dict(results)

    def get_result(self, embeds, path_dataset):
        if self.normalize:
            embeds.normalize()

        results = self.run(embeds, path_dataset)
        return results


class CosineDistance(SynonymyDetection):
    @classmethod
    def run_synonym_finding(self, embs, data):
        result = defaultdict(lambda: {})
        for word, suspicious_words in data.items():
            distances = []
            for susp_word, _ in suspicious_words:
                distances.append(1 - distance.cosine(embs.get_vector(susp_word), embs.get_vector(word)))
            guessed_word_index = distances.index(np.min(distances))
            results_for_word = []
            for dist_id, cosine_distance in enumerate(distances):
                d = {}
                d['suspicious_word'] = suspicious_words[dist_id][0]
                d['is_synonym'] = suspicious_words[dist_id][1]
                if dist_id == guessed_word_index:
                    d['hit'] = True
                else:
                    d['hit'] = False
                d['distance'] = cosine_distance
                results_for_word.append(d)
            result[word] = results_for_word
        return dict(result)
