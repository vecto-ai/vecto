from ..base import Benchmark
from collections import defaultdict
from sklearn import preprocessing
from sklearn.cluster import KMeans
from vecto.benchmarks.categorization.metrics import purity_score
from os import path, listdir
import csv

OTHER_EXT = 'None'
BENCHMARK = 'benchmark'

class Categorization(Benchmark):
    def __init__(self, normalize=True, ignore_oov=True):
        self.normalize = normalize
        self.ignore_oov = ignore_oov

    def read_test_set(self, path):
        test = defaultdict(lambda: [])
        if path.endswith('.csv'):
            with open(path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                head = True
                for row in reader:
                    if len(row) < 2:
                        continue
                    if not head:
                        category = row[1]
                        word = row[2]
                        test[category].append(word)
                    head = False
        else:
            with open(path) as f:
                for line in f:
                    id, category, word = line.strip().split()
                    test[category].append(word)
        return dict(test)

    def evaluate(self, embs, data):
        vectors = []
        labels = []

        for key, value in data.items():
            for word in value:
                if embs.has_word(word):
                    vectors.append(embs.get_vector(word))
                    labels.append(key)

        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        if len(data.keys()) > len(vectors):
            return None
        result = KMeans(n_clusters=len(data.keys()), random_state=10).fit_predict(vectors, labels)
        return purity_score(result, labels)

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
