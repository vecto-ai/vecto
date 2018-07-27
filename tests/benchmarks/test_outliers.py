"""Tests for outliers benchmark."""

import unittest
from io import StringIO
from contextlib import redirect_stdout
from vecto.benchmarks.outliers import *
from vecto.embeddings import load_from_dir
from ..test_setup import run_module
from numpy import array

path_categorization_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'outliers')


class Tests(unittest.TestCase):
    def test_categorization(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        categorization = AveragePairwiseCosine()
        result = categorization.get_result(embs, path_categorization_dataset)

    def test_cli(self):
        sio = StringIO()
        with redirect_stdout(sio):
            run_module('vecto.benchmarks.outliers',
                       './tests/data/embeddings/text/plain_with_file_header/',
                       './tests/data/benchmarks/outliers/',
                       '--path_out', '/tmp/vecto/', '--method', 'AveragePairwiseCosine')