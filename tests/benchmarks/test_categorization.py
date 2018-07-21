"""Tests for categorization benchmark."""

import unittest
from vecto.benchmarks.categorization import *
from vecto.embeddings import load_from_dir
from numpy import array

path_categorization_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'categorization')


class Tests(unittest.TestCase):
    def test_categorization(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        categorization = Categorization()
        result = categorization.get_result(embs, path_categorization_dataset)

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
