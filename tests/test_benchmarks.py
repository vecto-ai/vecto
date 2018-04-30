"""Tests for embeddings module."""

import unittest
from vecto.benchmarks.similarity import Similarity
from vecto.benchmarks.analogy import LRCos
from vecto.embeddings import load_from_dir


path_similarity_dataset = "./tests/data/benchmarks/similarity/"
path_analogy_dataset = "./tests/data/benchmarks/analogy/"

class Tests(unittest.TestCase):


    def test_similarity(self):
        embs = load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        similarity = Similarity()
        result = similarity.get_result(embs, path_similarity_dataset)
        print(result)

        similarity = Similarity(ignore_oov=False)
        result = similarity.get_result(embs, path_similarity_dataset)
        print(result)

    def test_analogy(self):
        embs = load_from_dir("./tests/data/embeddings/text/plain_with_file_header")
        analogy = LRCos()
        analogy.get_result(embs, path_analogy_dataset)


Tests().test_analogy()