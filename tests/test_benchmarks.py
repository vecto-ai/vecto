"""Tests for embeddings module."""

import unittest
from vecto.benchmarks.similarity.similarity import Similarity
from vecto.embeddings import load_from_dir


path_similarity_dataset = "./tests/data/benchmarks/similarity/"

class Tests(unittest.TestCase):


    def test_similarity(self):
        embs = load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        similarity = Similarity()
        result = similarity.get_result(embs, path_similarity_dataset)
        print(result)

        similarity = Similarity(ignore_oov=False)
        result = similarity.get_result(embs, path_similarity_dataset)
        print(result)
