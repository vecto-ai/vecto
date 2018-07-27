"""Tests for synonymy detection benchmark."""

import unittest
from io import StringIO
from contextlib import redirect_stdout
from vecto.benchmarks.synonymy_detection import *
from vecto.embeddings import load_from_dir
from ..test_setup import run_module

path_synonymy_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'synonymy_detection')


class Tests(unittest.TestCase):
    def test_synonymy(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        synonymy = CosineDistance()
        result = synonymy.get_result(embs, path_synonymy_dataset)
        print(result)

    def test_cli(self):
        sio = StringIO()
        with redirect_stdout(sio):
            run_module('vecto.benchmarks.synonymy_detection',
                       './tests/data/embeddings/text/plain_with_file_header/',
                       './tests/data/benchmarks/synonymy_detection',
                       '--path_out', '/tmp/vecto/', '--method', 'CosineDistance')
