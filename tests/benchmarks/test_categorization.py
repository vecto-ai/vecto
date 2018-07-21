"""Tests for categorization benchmark."""

import unittest
from io import StringIO
from contextlib import redirect_stdout
from vecto.benchmarks.categorization import *
from vecto.embeddings import load_from_dir
from ..test_setup import run_module
from numpy import array

path_categorization_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'categorization')


class Tests(unittest.TestCase):
    def test_categorization(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        categorization = Categorization()
        result = categorization.get_result(embs, path_categorization_dataset)

    def test_categorization_method(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        categorization = KMeansCategorization()
        result = categorization.get_result(embs, path_categorization_dataset)

        categorization = SpectralCategorization()
        result = categorization.get_result(embs, path_categorization_dataset)

    def test_cli(self):
        sio = StringIO()
        with redirect_stdout(sio):
            run_module('vecto.benchmarks.categorization',
                       './tests/data/embeddings/text/plain_with_file_header/',
                       './tests/data/benchmarks/categorization/',
                       '--path_out', '/tmp/vecto/', '--method', 'KMeansCategorization')

    # def test_cli_2(self):
    #     sio = StringIO()
    #     with redirect_stdout(sio):
    #         run_module('vecto.benchmarks.categorization',
    #                    './tests/data/embeddings/text/plain_with_file_header/',
    #                    './tests/data/benchmarks/categorization/',
    #                    '--path_out', '/tmp/vecto/r.json', '--method', 'SpectralCategorization')

    def test_set_loading(self):
        test_set_path = path.join('.', 'tests', 'data', 'benchmarks', 'categorization', 'essli-2008.csv')
        test_set_categories_amount = 9

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
