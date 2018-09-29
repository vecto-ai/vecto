from ..base import Benchmark
from collections import defaultdict
from os import path, listdir
import csv
import numpy as np
from scipy.spatial import distance
from itertools import product

OTHER_EXT = 'None'
BENCHMARK = 'benchmark'


class Outliers(Benchmark):
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
        test = defaultdict(lambda: [])
        if path.endswith('.csv'):
            with open(path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                head = True
                for row in reader:
                    if len(row) < 3:
                        continue
                    if not head:
                        category = row[1]
                        word = row[2]
                        is_outlier = row[3]
                        test[category].append({'word': word, 'is_outlier': is_outlier})
                    head = False
        else:
            with open(path) as f:
                for line in f:
                    _, category, word, is_outlier = line.strip().split()
                    test[category].append({'word': word, 'is_outlier': is_outlier})
        return dict(test)

    def collect_stats(self, categories):
        result = self.run_outliers(categories)
        return result

    def evaluate(self, embs, data):
        categories = defaultdict(lambda: defaultdict(lambda: []))
        for category, words in data.items():
            for value in words:
                if embs.has_word(value['word']):
                    categories[category]['words'].append([value['word'], embs.get_vector(value['word'])])
                    if value['is_outlier'] == 'true':
                        categories[category]['is_outlier'].append(True)
                    elif value['is_outlier'] == 'false':
                        categories[category]['is_outlier'].append(False)
                    else:
                        raise RuntimeError('Unexpected value occurred!')
        result = self.collect_stats(dict(categories))
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

    def run(self, embs, path_dataset):
        results = defaultdict(lambda: {})
        datasets = self.read_datasets_from_dir(path_dataset)
        for dataset_name, dataset_data in datasets.items():
            result = self.evaluate(embs, dataset_data)
            results[dataset_name] = result
        return dict(results)

    def get_result(self, embeds, path_dataset):
        if self.normalize:
            embeds.normalize()

        results = self.run(embeds, path_dataset)
        return results


class AveragePairwiseCosine(Outliers):
    def __init__(self, normalize=True,
                 ignore_oov=True,
                 do_top5=True,
                 need_subsample=False,
                 size_cv_test=1,
                 set_aprimes_test=None,
                 inverse_regularization_strength=1.0,
                 exclude=True,
                 threshold=0.5):
        self.threshold = threshold
        super().__init__(normalize,
                         ignore_oov,
                         do_top5,
                         need_subsample,
                         size_cv_test,
                         set_aprimes_test,
                         inverse_regularization_strength,
                         exclude)

    def run_outliers(self, categories):
        result = defaultdict(lambda: {})
        for category, values in categories.items():
            word_result = self.compare_words(values)
            result[category] = word_result
        return dict(result)

    @classmethod
    def compute_average(self, distances):
        return np.mean([value[1] for value in distances])

    def compare_words(self, values):
        result = defaultdict(lambda: {})
        distances_to_other_words = defaultdict(lambda: [])
        for word, compared_word in product(values['words'], repeat=2):
            if word == compared_word:
                continue
            distance_between_words = round(1 - distance.cosine(word[1], compared_word[1]), 2)
            distances_to_other_words[word[0]].append([compared_word[0], distance_between_words])
        for word_id, key in enumerate(distances_to_other_words.keys()):
            result_dict = {}
            result_dict['distances'] = distances_to_other_words[key]
            result_dict['is_outlier'] = values['is_outlier'][word_id]
            average = self.compute_average(distances_to_other_words[key])
            result_dict['average'] = round(average, 2)
            if average <= self.threshold:
                result_dict['hit'] = False
            else:
                result_dict['hit'] = True
            result[key] = result_dict
        return dict(result)
