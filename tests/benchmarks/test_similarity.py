"""Tests for analogy benchmark."""

import contextlib
import unittest
import io
from os import path
from vecto.benchmarks.similarity import Benchmark as Similarity
from vecto.benchmarks import visualize
from vecto.embeddings import load_from_dir
from vecto.data import Dataset
from tests.test_setup import run_module


path_similarity_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'similarity')
path_emb = path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header')


class Tests(unittest.TestCase):

    def test_api(self):
        embs = load_from_dir(path_emb)
        dataset = Dataset(path_similarity_dataset)
        similarity = Similarity()
        result = similarity.run(embs, dataset)
        self.assertIsInstance(result[0], dict)
        print(result)

        similarity = Similarity(ignore_oov=False)
        result = similarity.run(embs, dataset)
        self.assertIsInstance(result[0], dict)
        print(result)

        similarity = Similarity(normalize=False)
        result = similarity.run(embs, dataset)
        self.assertIsInstance(result[0], dict)
        print(result)

    def test_cli(self):
        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto",
                       "benchmark",
                       "similarity",
                       path_emb,
                       path_similarity_dataset,
                       "--path_out", "/tmp/vecto/benchmarks/")

        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto",
                       "benchmark",
                       "similarity",
                       path_emb,
                       path_similarity_dataset,
                       "--path_out", "/tmp/vecto/benchmarks/tmp")

        with self.assertRaises(FileNotFoundError):
            sio = io.StringIO()
            with contextlib.redirect_stdout(sio):
                run_module("vecto",
                           "benchmark",
                           "similarity",
                           path_emb + "NONEXISTING",
                           path_similarity_dataset,
                           "--path_out", "/tmp/vecto/benchmarks/")

        from matplotlib import pyplot as plt
        visualize.plot_accuracy("/tmp/vecto/benchmarks/word_similarity", key_secondary="experiment_setup.dataset")
        plt.savefig("/tmp/vecto/benchmarks/similarity.pdf", bbox_inches="tight")
