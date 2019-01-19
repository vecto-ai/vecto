"""Tests for embeddings module."""

import unittest
import os
import io
import contextlib
from tests.test_setup import run_module

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

    def test_abc(self):
        with self.assertRaises(NotImplementedError):
            vecto.benchmarks.base.Benchmark()
            # base.get_result(1, 2)

    def test_cli(self):
        with self.assertRaises(SystemExit):
            sio = io.StringIO()
            with contextlib.redirect_stdout(sio):
                run_module("vecto",
                           "benchmark",
                           "WRONG_NAME",
                           "path_embs")

        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto",
                       "benchmark",
                       "help")
