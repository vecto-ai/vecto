"""Tests for embeddings module."""

import unittest
from vecto.benchmarks.similarity import Similarity
from vecto.benchmarks.analogy import *
from vecto.embeddings import load_from_dir
from vecto.benchmarks.fetch_benchmarks import fetch_benchmarks
from os import path
from shutil import rmtree


path_similarity_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'similarity')
path_analogy_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'analogy')


class Tests(unittest.TestCase):
    def test_similarity(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        similarity = Similarity()
        result = similarity.get_result(embs, path_similarity_dataset)
        print(result)

        similarity = Similarity(ignore_oov=False)
        result = similarity.get_result(embs, path_similarity_dataset)
        print(result)

    def test_analogy(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        analogy = LinearOffset()
        result = analogy.get_result(embs, path_analogy_dataset)
        print(result)

        analogy = PairDistance()
        result = analogy.get_result(embs, path_analogy_dataset)
        print(result)

        analogy = ThreeCosMul()
        result = analogy.get_result(embs, path_analogy_dataset)
        print(result)

        analogy = ThreeCosMul2()
        result = analogy.get_result(embs, path_analogy_dataset)
        print(result)
        # analogy = SimilarToAny()
        # result = analogy.get_result(embs, path_analogy_dataset)
        # print(result)
        # analogy = SimilarToB()
        # result = analogy.get_result(embs, path_analogy_dataset)
        # print(result)
        analogy = LRCos()
        result = analogy.get_result(embs, path_analogy_dataset)
        print(result)

    def test_fetcher(self):
        if path.isdir(path.join('.', 'tests', 'data', 'benchmarks_test')):
            return
        fetch_benchmarks(path.join('.', 'tests', 'data', 'benchmarks_test'))
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        similarity = Similarity()
        path_similarity_dataset = path.join('.', 'tests', 'data', 'benchmarks_test', 'benchmarks', 'similarity', 'en')
        result = similarity.get_result(embs, path_similarity_dataset),


        # big embs and dataset test
        # embs = load_from_dir("/home/bofang/Documents/embeddings/negative_sampling/fair/")
        # result = analogy.get_result(embs, "/home/bofang/Downloads/BATS_3.0_small")
        # print(result)




# Tests().test_analogy()