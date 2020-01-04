"""Tests for categorization benchmark."""

import unittest
from io import StringIO
from contextlib import redirect_stdout
from vecto.benchmarks.categorization import Benchmark as Categorization
from vecto.benchmarks.categorization import purity_score
from vecto.embeddings import load_from_dir
from ..test_setup import run_module
from numpy import array
from os import path

path_categorization_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'categorization')


class Tests(unittest.TestCase):
    # def test_categorization(self):
    #     embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
    #     categorization = KMeansCategorization()
    #     result = categorization.get_result(embs, path_categorization_dataset)

    # def test_categorization_method_works(self):
    #     embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
    #     categorization = KMeansCategorization()
    #     result = categorization.get_result(embs, path_categorization_dataset)

    def test_cli(self):
        sio = StringIO()
        with redirect_stdout(sio):
            run_module('vecto',
                       'benchmark',
                       'categorization',
                       './tests/data/embeddings/text/plain_with_file_header/',
                       './tests/data/benchmarks/categorization/',
                       '--path_out', '/tmp/vecto/benchmarks')

    # def test_categorization_scores(self):
    #     embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
    #     categorization = KMeansCategorization()
    #     result = categorization.get_result(embs, path_categorization_dataset)
    #     scores = result[0]['global_stats']['scores']
    #     self.assertEqual(len(scores.keys()), 7)
    #     self.assertEqual(len(result[0]['global_stats']['true_labels']), 7)
    #     self.assertEqual(result[0]['global_stats']['true_labels'][3], 1)

    # def test_categorization_data(self):
    #     embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
    #     categorization = KMeansCategorization()
    #     result = categorization.get_result(embs, path_categorization_dataset)
    #     word_stats = result[0]['word_stats']
    #     # self.assertEqual(word_stats['4. banana']['true_category'], 'food')
    #     self.assertEqual(len(word_stats.keys()), 7)

    # def test_kmeans(self):
    #     data = [(0, 0, 0), (100, 100, 100), (99, 99, 99)]
    #     keys_len = 2
    #     labels = [0, 1]
    #     categorization = KMeansCategorization()
    #     predicted_labels, true_labels, centroids, inertia, params = categorization.run_categorization(keys_len, data, labels)
    #     self.assertEqual(len(centroids), 2)
    #     self.assertEqual(inertia, 1.5)

    # def test_cli_2(self):
    #     sio = StringIO()
    #     with redirect_stdout(sio):
    #         run_module('vecto.benchmarks.categorization',
    #                    './tests/data/embeddings/text/plain_with_file_header/',
    #                    './tests/data/benchmarks/categorization/',
    #                    '--path_out', '/tmp/vecto/r.json', '--method', 'SpectralCategorization')

    def test_set_loading(self):
        test_set_path = path.join('.', 'tests', 'data', 'benchmarks', 'categorization', 'essli-2008-lite.csv')
        test_set_categories_amount = 3

        categorization = Categorization()
        test_set = categorization.read_test_set(test_set_path)
        self.assertEqual(len(test_set.keys()), test_set_categories_amount)

    def test_purity_measure(self):
        test_set_1 = array((0, 1, 2, 3))
        test_set_2 = array((0, 1, 2, 3))
        expected_score = 1.0
        self.assertEqual(purity_score(test_set_1, test_set_2), expected_score)

        test_set_1 = array((0, 0, 3, 3))
        test_set_2 = array((0, 0, 0, 0))
        expected_score = 0.5
        self.assertEqual(purity_score(test_set_1, test_set_2), expected_score)
