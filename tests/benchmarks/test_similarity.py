"""Tests for similarity benchmark."""

import unittest
from os import path
from vecto.benchmarks.similarity import similarity, visualize
from vecto.embeddings import load_from_dir

path_similarity_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'similarity')


class Tests(unittest.TestCase):
    def test_similarity(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        sim = similarity.SpearmanCorrelation()
        result = sim.get_result(embs, path_similarity_dataset)
        self.assertEqual(result[0]['Experiment_setup']['Cnt_pairs_total'], 20)
        self.assertEqual(result[0]['Result'], None)
        print(result)

    def test_similarity_without_oov_ignore(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        sim = similarity.SpearmanCorrelation(ignore_oov=False)
        result = sim.get_result(embs, path_similarity_dataset)
        self.assertEqual(result[0]['Experiment_setup']['Cnt_pairs_total'], 20)
        print(result)

    @classmethod
    def test_similarity_visualization(cls):
        visualize.plot_accuracy(path.join('tests' 'data', 'benchmarks_results', 'similarity'))
