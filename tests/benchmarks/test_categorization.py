"""Tests for categorization benchmark."""

import contextlib
import unittest
import sys
import io
from os import path
from vecto.benchmarks.categorization import *
from vecto.benchmarks.analogy import visualize as analogy_visualize
from vecto.embeddings import load_from_dir
from ..test_setup import run_module


path_categorization_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'categorization')

class Tests(unittest.TestCase):
    def test_categorization(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        categorization = Categorization()
        result = categorization.get_result(embs, path_categorization_dataset)
        print(result)