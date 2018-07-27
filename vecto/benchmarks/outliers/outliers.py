from ..base import Benchmark
from collections import defaultdict
from sklearn import preprocessing
from os import path, listdir
import csv
import numpy as np
from scipy.spatial import distance

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

    def read_test_set(self, path):
        test = defaultdict(lambda: {})
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
                        test[category] = {'word': word, 'is_outlier': is_outlier}
                    head = False
        else:
            with open(path) as f:
                for line in f:
                    id, category, word, is_outlier = line.strip().split()
                    test[category] = {'word': word, 'is_outlier': is_outlier}
        return dict(test)

    def collect_stats(self, data, vectors, labels):
        pass

    def evaluate(self, embs, data):
        categories = []
        vectors = []
        labels = []
        for category, value in data.items():
            if embs.has_word(value['word']):
                vectors.append(embs.get_vector(value['word']))
                labels.append(value['is_outlier'])
                categories.append(category)
        self.collect_stats(vectors, categories, labels)
        result = {}
        return result

    def read_datasets_from_dir(self, path_to_dir):
        datasets = defaultdict(lambda: {})
        for file in listdir(path_to_dir):
            dataset_name, dataset_data = self.read_single_dataset(path_to_dir, file)
            if type != OTHER_EXT:
                datasets[dataset_name] = dataset_data
        return datasets

    def read_single_dataset(self, path_to_dir, file_name):
        dataset_name, file_extension = path.splitext(file_name)
        data = self.read_test_set(path.join(path_to_dir, file_name))
        return dataset_name, data

    def run(self, embs, path_dataset):
        results = []
        datasets = self.read_datasets_from_dir(path_dataset)
        for dataset_name, dataset_data in datasets.items():
            result = self.evaluate(embs, dataset_data)
            results.append(result)
        return results

    def get_result(self, embs, path_dataset):
        if self.normalize:
            embs.normalize()

        results = self.run(embs, path_dataset)
        return results


class Pairwise(Outliers):
    def run_outliers(self, vectors, categories, labels):
        pass
