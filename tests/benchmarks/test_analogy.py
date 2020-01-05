"""Tests for analogy benchmark."""

import contextlib
import unittest
import io
from os import path
from vecto.benchmarks.analogy import Benchmark as Analogy
from vecto.benchmarks import visualize
from vecto.embeddings import load_from_dir
from vecto.data import Dataset

from ..test_setup import run_module


path_analogy_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'analogy')


class Tests(unittest.TestCase):

    def test_api(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        analogy = Analogy(method="3CosAdd")
        dateset = Dataset(path_analogy_dataset)
        result = analogy.run(embs, dateset)
        self.assertIsInstance(result[0], dict)

        analogy = Analogy(method="PairDistance")
        result = analogy.run(embs, dateset)
        self.assertIsInstance(result[0], dict)

        analogy = Analogy(method="3CosMul")
        result = analogy.run(embs, dateset)
        self.assertIsInstance(result[0], dict)

        analogy = Analogy(method="3CosMul2")
        result = analogy.run(embs, dateset)
        self.assertIsInstance(result[0], dict)

        analogy = Analogy(method="3CosAvg")
        result = analogy.run(embs, dateset)
        self.assertIsInstance(result[0], dict)

        analogy = Analogy(method="SimilarToAny")
        result = analogy.run(embs, dateset)
        print(result)

        analogy = Analogy(method="SimilarToB")
        result = analogy.run(embs, dateset)
        print(result)

        analogy = Analogy(method="LRCos")
        result = analogy.run(embs, dateset)
        print(result)

    def test_cli(self):
        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto", "benchmark", "analogy",
                       "./tests/data/embeddings/text/plain_with_file_header/",
                       "./tests/data/benchmarks/analogy/",
                       "--path_out", "/tmp/vecto/benchmarks/",
                       "--method", "3CosAdd")

        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto", "benchmark", "analogy",
                       "./tests/data/embeddings/text/plain_with_file_header/",
                       "./tests/data/benchmarks/analogy/",
                       "--path_out",
                       "/tmp/vecto/benchmarks/specific_filename.json",
                       "--method", "LRCos")

        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto", "benchmark", "analogy",
                       "./tests/data/embeddings/text/plain_with_file_header/",
                       "./tests/data/benchmarks/analogy/",
                       "--path_out", "/tmp/vecto/benchmarks/",
                       "--method", "3CosMul")

        sio = io.StringIO()
        with self.assertRaises(RuntimeError):
            with contextlib.redirect_stdout(sio):
                run_module("vecto", "benchmark", "analogy",
                           "./tests/data/embeddings/text/plain_with_file_header/",
                           "./tests/data/benchmarks/analogy/",
                           "--method", "NONEXISTING")

        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto", "benchmark", "analogy",
                       "./tests/data/embeddings/text/plain_with_file_header/",
                       "./tests/data/benchmarks/analogy/",
                       "--method", "3CosAvg")

        # TODO: suppress concatenating timestamp or aggregate multiple runs
        from matplotlib import pyplot as plt
        visualize.plot_accuracy("/tmp/vecto/benchmarks/word_analogy")
        plt.savefig("/tmp/vecto/benchmarks/analogy.pdf", bbox_inches="tight")
