"""Tests for synonymy detection benchmark."""

import unittest
from io import StringIO
from contextlib import redirect_stdout
from vecto.benchmarks.synonymy_detection import *
from vecto.embeddings import load_from_dir
from ..test_setup import run_module

path_synonymy_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'synonymy_detection')


class Tests(unittest.TestCase):
    @classmethod
    def test_synonymy(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        synonymy = CosineDistance()
        synonymy.get_result(embs, path_synonymy_dataset)

    @classmethod
    def test_cli(self):
        sio = StringIO()
        with redirect_stdout(sio):
            run_module('vecto.benchmarks.synonymy_detection',
                       './tests/data/embeddings/text/plain_with_file_header/',
                       './tests/data/benchmarks/synonymy_detection',
                       '--path_out', '/tmp/vecto/benchmarks', '--method', 'CosineDistance')

    def test_synonymy_results(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        synonymy = CosineDistance()
        result = synonymy.get_result(embs, path_synonymy_dataset)['test']
        cat_is_synonym = 'yes'
        cat_is_hit = False
        distance_to_cat = 1.0

        self.assertEqual(result['tiger'][0]['is_synonym'], cat_is_synonym)
        self.assertEqual(result['tiger'][0]['hit'], cat_is_hit)
        self.assertEqual(result['tiger'][0]['distance'], distance_to_cat)

    def test_synonymy_reader(self):
        synonymy = CosineDistance()
        test_set = synonymy.read_test_set(path.join(path_synonymy_dataset, 'test.csv'))
        expected_amount_of_keys = 2
        expected_amount_of_tiger_suspicious = 3
        cat_is_synonym_with_tiger = 'yes'

        self.assertEqual(len(test_set.keys()), expected_amount_of_keys)
        self.assertEqual(len(test_set['tiger']), expected_amount_of_tiger_suspicious)
        self.assertEqual(test_set['tiger'][0][1], cat_is_synonym_with_tiger)
