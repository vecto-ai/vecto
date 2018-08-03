"""Tests for analogy benchmark."""

import contextlib
import unittest
import io
from os import path
from vecto.benchmarks.similarity import *
from vecto.benchmarks import visualize
from vecto.embeddings import load_from_dir
from tests.test_setup import run_module

path_similarity_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'similarity')
path_emb = path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header')


class Tests(unittest.TestCase):

    def test_api(self):
        embs = load_from_dir(path_emb)
        similarity = Similarity()
        result = similarity.get_result(embs, path_similarity_dataset)
        self.assertIsInstance(result[0], dict)
        print(result)

        similarity = Similarity(ignore_oov=False)
        result = similarity.get_result(embs, path_similarity_dataset)
        self.assertIsInstance(result[0], dict)
        print(result)

        similarity = Similarity(normalize=False)
        result = similarity.get_result(embs, path_similarity_dataset)
        self.assertIsInstance(result[0], dict)
        print(result)

    def test_cli(self):
        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto.benchmarks.similarity",
                       path_emb,
                       path_similarity_dataset,
                       "--path_out", "/tmp/vecto/benchmarks/")

        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto.benchmarks.similarity",
                       path_emb,
                       path_similarity_dataset,
                       "--path_out", "/tmp/vecto/benchmarks/tmp")

        with self.assertRaises(FileNotFoundError):
            sio = io.StringIO()
            with contextlib.redirect_stdout(sio):
                run_module("vecto.benchmarks.similarity",
                           path_emb + "NONEXISTING",
                           path_similarity_dataset,
                           "--path_out", "/tmp/vecto/benchmarks/")

#         from matplotlib import pyplot as plt
#         visualize.plot_accuracy("/tmp/vecto/benchmarks/similarity", key_secondary=None)
#         plt.savefig("/tmp/vecto/benchmarks/similarity.pdf", bbox_inches="tight")
#
#
# Tests().test_cli()
