"""Tests for embeddings module."""

import unittest
from vecto.embeddings.dense import WordEmbeddingsDense


class Tests(unittest.TestCase):

    def test_basic(self):
        WordEmbeddingsDense()
