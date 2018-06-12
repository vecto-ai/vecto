"""Tests for embeddings module."""

import unittest
from os import path

from vecto.benchmarks import text_classification
from vecto.benchmarks.similarity import Similarity
from vecto.benchmarks.similarity import visualize as similarity_visualize
from vecto.benchmarks.text_classification import Text_classification
from vecto.embeddings import load_from_dir
from vecto.utils.fetch_benchmarks import fetch_benchmarks
from os import path
from shutil import rmtree


path_similarity_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'similarity')
path_text_classification_dataset = path.join('.', 'tests', 'data', 'benchmarks', 'text_classification')


class Tests(unittest.TestCase):
    def test_similarity(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        similarity = Similarity()
        result = similarity.get_result(embs, path_similarity_dataset)
        print(result)

        similarity = Similarity(ignore_oov=False)
        result = similarity.get_result(embs, path_similarity_dataset)
        print(result)

        similarity_visualize.plot_accuracy()

    def test_fetcher(self):
        if path.isdir(path.join('.', 'tests', 'data', 'benchmarks_test')):
            return
        fetch_benchmarks(path.join('.', 'tests', 'data', 'benchmarks_test'))
#         embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
#         similarity = Similarity()
#         path_similarity_dataset = path.join('.', 'tests', 'data', 'benchmarks_test', 'benchmarks', 'similarity', 'en')
#         result = similarity.get_result(embs, path_similarity_dataset),

        # big embs and dataset test
        # embs = load_from_dir("/home/bofang/Documents/embeddings/negative_sampling/fair/")
        # result = analogy.get_result(embs, "/home/bofang/Downloads/BATS_3.0_small")
        # print(result)

    def test_text_classification(self):
        embs = load_from_dir("./tests/data/embeddings/text/plain_with_file_header")

        tc = Text_classification(model='cnn')
        result = tc.get_result(embs, path_text_classification_dataset,
                               "/tmp/tests/data/benchmarks_results/text_classification/")
        print(result)
        tc = Text_classification(model='rnn')
        result = tc.get_result(embs, path_text_classification_dataset,
                               "/tmp/tests/data/benchmarks_results/text_classification/")
        print(result)
        tc = Text_classification(model='bow')
        result = tc.get_result(embs, path_text_classification_dataset,
                               "/tmp/tests/data/benchmarks_results/text_classification/")
        print(result)

        model = text_classification.load_model("./tests/data/benchmarks_results/text_classification/args.json", embs.matrix)
        print(text_classification.predict(model, "I like this"))
        print(text_classification.get_vectors(model, ["I like this", "I hate this"]))
