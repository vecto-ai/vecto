"""Tests for embeddings module."""

import unittest
import numpy as np
from vecto.embeddings.dense import WordEmbeddingsDense
from vecto.embeddings import load_from_dir


class Tests(unittest.TestCase):

    def test_basic(self):
        WordEmbeddingsDense()
        model = load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        model.cmp_words("apple", "banana")
        model.cmp_words("apple", "bananaaaaa")
        x = np.array([0.0, 0.0, 0.0])
        x.fill(np.nan)
        model.cmp_vectors(x, x)

    def test_load(self):
        load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        # TODO: assert right class
        load_from_dir("tests/data/embeddings/text/plain_no_file_header")
        # TODO: assert right class
        load_from_dir("tests/data/embeddings/npy")

        embs = load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        embs.get_vector('apple')
        with self.assertRaises(RuntimeError):
            embs.get_vector('word_that_not_in_vocabulary_27')

    def test_utils(self):
        embs = load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        results = embs.get_most_similar_words('apple', 5)
        print(results)
        embs.cache_normalized_copy()
        results = embs.get_most_similar_words('apple', 5)
        print(results)

        results = embs.get_most_similar_words(embs.get_vector('apple'), 5)
        print(results)

    def test_save(self):
        embs = load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        path_save = "/tmp/vecto/saved"
        embs.save_to_dir(path_save)
        embs = load_from_dir(path_save)
        print(embs.matrix.shape)
        embs.save_to_dir_plain_txt("/tmp/vecto/saved_plain")
