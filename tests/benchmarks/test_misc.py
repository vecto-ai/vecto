"""Tests for embeddings module."""

import unittest
import os

import vecto
from vecto.benchmarks import text_classification
from vecto.benchmarks.similarity import Similarity
from vecto.benchmarks.sequence_labeling import Sequence_labeling
from vecto.benchmarks.language_modeling import Language_modeling
# from vecto.benchmarks.similarity import visualize as similarity_visualize
from vecto.benchmarks.text_classification import Text_classification
from vecto.embeddings import load_from_dir
from vecto.utils.fetch_benchmarks import fetch_benchmarks
from os import path

# from shutil import rmtree

path_similarity_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'similarity')
path_sequence_labeling_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'sequence_labeling')
path_language_modeling_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'language_modeling')


class Tests(unittest.TestCase):

    def test_fetcher(self):
        if path.isdir(path.join('.', 'tests', 'data', 'benchmarks_test')):
            return
        fetch_benchmarks(path.join('.', 'tests', 'data', 'benchmarks_test'))
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        similarity = Similarity()
        path_similarity_dataset = path.join('.', 'tests', 'data', 'benchmarks_test', 'benchmarks', 'similarity', 'en')
        similarity.get_result(embs, path_similarity_dataset)

    def test_sequence_labeling(self):
        embs = load_from_dir("./tests/data/embeddings/text/plain_with_file_header")

        for method in ['lr', '2FFNN']:
            sequence_labeling = Sequence_labeling(method='lr')
            for subtask in ['chunk', 'pos', 'ner']:  # , 'chunk', 'pos', 'ner'
                results = sequence_labeling.get_result(embs, os.path.join(path_sequence_labeling_dataset, subtask))
                print(results)

    def test_abc(self):
        with self.assertRaises(NotImplementedError):
            vecto.benchmarks.base.Benchmark()
            # base.get_result(1, 2)
