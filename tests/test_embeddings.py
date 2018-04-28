"""Tests for embeddings module."""

import unittest
from vecto.embeddings.dense import WordEmbeddingsDense
from vecto.embeddings import load_from_dir


class Tests(unittest.TestCase):

    def test_basic(self):
        WordEmbeddingsDense()

    def test_load(self):
        load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        # TODO: assert right class
        load_from_dir("tests/data/embeddings/text/plain_no_file_header")
        # TODO: assert right class
        load_from_dir("tests/data/embeddings/npy")
