from ..base import Benchmark
from vecto._version import VERSION
from collections import defaultdict
# from sklearn import preprocessing
from sklearn.cluster import KMeans
from vecto.benchmarks.categorization.metrics import *
from os import path, listdir
import csv
# import numpy as np
from scipy.spatial import distance
import os

OTHER_EXT = 'None'
BENCHMARK = 'benchmark'


class Categorization(Benchmark):
    def __init__(self, normalize=True,
                 ignore_oov=True,
                 do_top5=True,
                 need_subsample=False,
                 size_cv_test=1,
                 set_aprimes_test=None,
                 inverse_regularization_strength=1.0,
                 exclude=True,
                 random_state=10):
        self.normalize = normalize
        self.ignore_oov = ignore_oov
        self.do_top5 = do_top5
        self.need_subsample = need_subsample
        self.normalize = normalize
        self.size_cv_test = size_cv_test
        self.set_aprimes_test = set_aprimes_test
        self.inverse_regularization_strength = inverse_regularization_strength
        self.exclude = exclude
        self.random_state = random_state

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

    def round_scores(self, scores, after_comma=2):
        new_items = {}
        for key, score in scores.items():
            new_items[key] = round(score, after_comma)
        return new_items

    def compute_metics(self, predicted_labels, true_labels):
        results = {}
        results['Purity'] = purity_score(predicted_labels, true_labels)
        results['Ari'] = adjusted_rand_score(predicted_labels, true_labels)
        results['Homogeneity'] = homogeneity_score(predicted_labels, true_labels)
        results['Completeness'] = completeness_score(predicted_labels, true_labels)
        results['V-measure'] = v_measure_score(predicted_labels, true_labels)
        results['Mutual info'] = mutual_info_score(predicted_labels, true_labels)
        results['Fowlkes-Mallows'] = fowlkes_mallows_score(predicted_labels, true_labels)
        results = self.round_scores(results)
        return results

    def evaluate(self, embs, data):
        vectors = []
        labels = []
        new_data = defaultdict(lambda: [])
        for key, value in data.items():
            for word in value:
                if embs.has_word(word):
                    vectors.append(embs.get_vector(word))
                    labels.append(list(data.keys()).index(key))
                    new_data[key].append(word)
        if len(data.keys()) > len(vectors):
            raise Exception('Too poor vocabulary')
        word_stats, global_stats = self.collect_stats(new_data, vectors, labels)
        result = {}
        result['word_stats'] = word_stats
        result['global_stats'] = global_stats

        # add experiment_setup and result entry for result
        result["experiment_setup"] = {}
        result["result"] = result['global_stats']['scores']
        result["experiment_setup"]['default_measurement'] = 'Purity'
        result["experiment_setup"]["task"] = "categorization"

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

    def run(self, embs, dataset):
        path_dataset = dataset.path
        if self.normalize:
            embs.normalize()
        results = []
        datasets = self.read_datasets_from_dir(path_dataset)
        for dataset_name, dataset_data in datasets.items():
            result = self.evaluate(embs, dataset_data)
            result['experiment_setup']['dataset'] = os.path.basename(os.path.normpath(path_dataset))
            result['experiment_setup']['embeddings'] = embs.metadata
            result['experiment_setup']['method'] = self.method
            result['experiment_setup']['vecto_version'] = VERSION
            results.append(result)
        return results

    def collect_stats(self, data, vectors, labels):
        word_stats = defaultdict(lambda: {})
        predicted_labels, true_labels, centroids, inertia, params = self.run_categorization(len(data.keys()), vectors,
                                                                                            labels)
        categories = data.keys()
        word_counter = 0
        for category, words in data.items():
            for word_id, word in enumerate(words):
                word_vector = vectors[word_counter]
                centroid = centroids[predicted_labels[word_counter]]
                predicted_category = list(categories)[predicted_labels[word_counter]]
                word_entry = '{}. {}'.format(word_counter, word)
                word_counter += 1
                word_stats[word_entry] = self.process_stats(word_vector, centroid, category, predicted_category)
        metric_scores = self.compute_metics(predicted_labels, true_labels)
        global_stats = self.process_global_stats(inertia, params, metric_scores, categories, predicted_labels,
                                                 true_labels)
        return dict(word_stats), global_stats


class KMeansCategorization(Categorization):
    def run_categorization(self, clusters_amount, vectors, true_labels):
        kmeans = KMeans(n_clusters=clusters_amount,
                        init='k-means++',
                        n_init=10,
                        max_iter=50,
                        tol=0.0001,
                        precompute_distances='auto',
                        verbose=0,
                        random_state=None,
                        copy_x=True,
                        n_jobs=1,
                        algorithm='auto')
        predicted_labels = kmeans.fit_predict(vectors, true_labels)
        centroids = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        params = str(kmeans.get_params())
        return predicted_labels, true_labels, centroids, inertia, params

    def process_stats(self, word_vector, centroid, category, predicted_category):
        stats = {}
        if category == predicted_category:
            hit = 'true'
        else:
            hit = 'false'
        stats['true_category'] = category
        stats['predicted_category'] = predicted_category
        stats['hit'] = hit
        stats['distance_to_centroid'] = 1 - distance.cosine(word_vector, centroid)
        return stats

    def process_global_stats(self, inertia, params, metric_scores, categories, predicted_labels, true_labels):
        global_stats = {}
        global_stats['inertia'] = float(inertia)
        global_stats['params'] = params
        global_stats['scores'] = metric_scores
        global_stats['categories'] = list(categories)
        global_stats['predicted_labels'] = list(int(label) for label in predicted_labels)
        global_stats['true_labels'] = list(int(label) for label in true_labels)
        return global_stats

# class SpectralCategorization(Categorization):
#     def compute_labels(self, data, vectors, labels):
#         return SpectralClustering(n_clusters=len(data.keys()),
#                                   random_state=self.random_state).fit_predict(vectors,
#                                                                                                            labels)
