"""Tests for embeddings module."""

import unittest
from os import path

from vecto.benchmarks.analogy import *
from vecto.benchmarks.analogy import visualize as analogy_visualize
from vecto.embeddings import load_from_dir
from os import path


path_analogy_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'analogy')


class Tests(unittest.TestCase):

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

        analogy = ThreeCosAvg()
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
        analogy_visualize.plot_accuracy()

        # big embs and dataset test
        # embs = load_from_dir("/home/bofang/Documents/embeddings/negative_sampling/fair/")
        # result = analogy.get_result(embs, "/home/bofang/Downloads/BATS_3.0_small")
        # print(result)


