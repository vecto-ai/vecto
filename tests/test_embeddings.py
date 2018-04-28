"""Tests for embeddings module."""

import unittest
from vecto.embeddings.dense import WordEmbeddingsDense
from vecto.embeddings import load_from_dir


class Tests(unittest.TestCase):

    def test_basic(self):
        WordEmbeddingsDense()
