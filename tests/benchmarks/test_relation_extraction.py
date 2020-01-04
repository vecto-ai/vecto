"""Tests for analogy benchmark."""

import contextlib
import unittest
import io
from os import path
from vecto.benchmarks import visualize
from vecto.embeddings import load_from_dir
from vecto.data import Dataset
from tests.test_setup import run_module


path_emb = path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header')
path_dataset = path.join('tests', 'data', 'benchmarks', 'relation_extraction')


class Tests(unittest.TestCase):
    # def test_api(self):
    #     embs = load_from_dir(path_emb)

    #     for method in ['lr', '2FFNN']:
    #         sequence_labeling = Sequence_labeling(method=method)
    #         for subtask in ['chunk', 'pos', 'ner']:  # , 'chunk', 'pos', 'ner'
    #             result = sequence_labeling.get_result(embs, path.join(path_sequence_labeling_dataset, subtask))
    #             self.assertIsInstance(result[0], dict)
    #             print(result)

    def test_cli(self):
        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto",
                       "benchmark",
                       "relation_extraction",
                       path_emb,
                       path_dataset,
                       "--path_out", "/tmp/vecto/benchmarks/")

        with self.assertRaises(FileNotFoundError):
            sio = io.StringIO()
            with contextlib.redirect_stdout(sio):
                run_module("vecto",
                           "benchmark",
                           "relation_extraction",
                           path_emb + "NONEXISTING",
                           path_dataset,
                           "--path_out",
                           "/tmp/vecto/benchmarks/")

        from matplotlib import pyplot as plt
        visualize.plot_accuracy("/tmp/vecto/benchmarks/relation_extraction", key_secondary="experiment_setup.dataset")
        plt.savefig("/tmp/vecto/benchmarks/relation_extraction.pdf", bbox_inches="tight")
