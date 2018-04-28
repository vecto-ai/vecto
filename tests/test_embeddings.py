"""Tests for embeddings module."""

import unittest
from vecto.embeddings.dense import WordEmbeddingsDense
from vecto.embeddings import load_from_dir


class Tests(unittest.TestCase):

    def test_basic(self):
        WordEmbeddingsDense()

    def test_load_(self):
        load_from_dir("tests/data/embeddings/text/plain_with_file_header")
