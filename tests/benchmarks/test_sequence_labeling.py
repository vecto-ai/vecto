"""Tests for analogy benchmark."""

import contextlib
import unittest
import io
from os import path
from vecto.benchmarks.sequence_labeling import Benchmark as Sequence_labeling
from vecto.benchmarks import visualize
from vecto.embeddings import load_from_dir
from vecto.data import Dataset
from tests.test_setup import run_module


path_sequence_labeling_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'sequence_labeling')
path_sequence_labeling_dataset_ner = path.join('.', 'tests', 'data', 'benchmarks', 'sequence_labeling', 'ner') # sequence labeling need to specify a sub task (pos, chunk, or ner)
path_emb = path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header')


class Tests(unittest.TestCase):
    def test_api(self):
        embs = load_from_dir(path_emb)

        for method in ['lr', '2FFNN']:
            sequence_labeling = Sequence_labeling(method=method)
            for subtask in ['chunk', 'pos', 'ner']:  # , 'chunk', 'pos', 'ner'
                dataset = Dataset(path.join(path_sequence_labeling_dataset, subtask))
                result = sequence_labeling.run(embs, dataset)
                self.assertIsInstance(result[0], dict)
                print(result)

    def test_cli(self):
        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto",
                       "benchmark",
                       "sequence_labeling",
                       path_emb,
                       path_sequence_labeling_dataset_ner,
                       "--path_out", "/tmp/vecto/benchmarks/")

        sio = io.StringIO()
        with contextlib.redirect_stdout(sio):
            run_module("vecto",
                       "benchmark",
                       "sequence_labeling",
                       path_emb,
                       path_sequence_labeling_dataset_ner,
                       "--path_out", "/tmp/vecto/benchmarks/")

        with self.assertRaises(FileNotFoundError):
            sio = io.StringIO()
            with contextlib.redirect_stdout(sio):
                run_module("vecto",
                           "benchmark",
                           "sequence_labeling",
                           path_emb + "NONEXISTING",
                           path_sequence_labeling_dataset_ner,
                           "--path_out",
                           "/tmp/vecto/benchmarks/")

        from matplotlib import pyplot as plt
        # here the visualization only for the ner sub task.
        visualize.plot_accuracy("/tmp/vecto/benchmarks/sequence_labeling/ner", key_secondary="experiment_setup.dataset")
        plt.savefig("/tmp/vecto/benchmarks/sequence_labeling.pdf", bbox_inches="tight")
