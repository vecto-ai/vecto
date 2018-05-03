"""Tests for embeddings module."""

import unittest
import numpy as np
from scipy.spatial.distance import cosine
from vecto.embeddings import load_from_dir
from vecto.vocabulary import Vocabulary
from vecto.embeddings.composite import ArithmeticMeanVector, GeometricMeanVector, \
    IDFArithmeticMeanVector, IDFGeometricMeanVector
from vecto.utils.blas import normed


def make_dummy_vocab():
    result = Vocabulary()
    result.lst_words = ['the', 'apple', 'banana', 'fast', 'quick', 'tiger', 'cat']
    result.lst_frequencies = [150, 2, 2, 20, 2, 5, 2]
    result.dic_words_ids = { w : i for i, w in enumerate(result.lst_words) }
    return result


class Tests(unittest.TestCase):

    def test_arif_avg(self):
        base_emb = load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        comp = ArithmeticMeanVector(base_emb)
        assert comp.dimensions_number == 4

        gold = (base_emb.get_vector('banana') + base_emb.get_vector('tiger')) / 2
        assert np.allclose(comp.get_vector(['banana', 'tiger']), gold)

    def test_arif_avg_l2norm(self):
        base_emb = load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        comp = ArithmeticMeanVector(base_emb, norm=None)
        assert comp.dimensions_number == 4

        gold = (base_emb.get_vector('banana') + base_emb.get_vector('tiger')) / 2
        gold = normed(gold, ord=2)
        assert np.allclose(comp.get_vector(['banana', 'tiger']), gold)

    def test_geom_avg(self):
        base_emb = load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        comp = GeometricMeanVector(base_emb)
        assert comp.dimensions_number == 4

        gold = (base_emb.get_vector('banana') * base_emb.get_vector('tiger')) ** 0.5
        assert np.allclose(comp.get_vector(['banana', 'tiger']), gold)

    def test_idf_arif_avg(self):
        base_emb = load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        vocab_with_freq = make_dummy_vocab()  # load any vocab with non-zero frequencies
        comp = IDFArithmeticMeanVector(base_emb, vocab=vocab_with_freq)
        assert comp.dimensions_number == 4

        apple = comp.get_vector(['apple'])
        the_apple = comp.get_vector(['the', 'apple'])
        quick_apple = comp.get_vector(['quick', 'apple'])
        the_quick_apple = comp.get_vector(['the', 'quick', 'apple'])
        the_fast_apple = comp.get_vector(['the', 'fast', 'apple'])

        assert cosine(apple, the_apple) < cosine(apple, quick_apple)
        assert cosine(apple, quick_apple) < cosine(apple, the_quick_apple)
        assert cosine(apple, the_fast_apple) < cosine(apple, the_quick_apple)

    def test_idf_geom_avg(self):
        base_emb = load_from_dir("tests/data/embeddings/text/plain_with_file_header")
        vocab_with_freq = make_dummy_vocab()  # load any vocab with non-zero frequencies
        comp = IDFGeometricMeanVector(base_emb, vocab=vocab_with_freq)
        assert comp.dimensions_number == 4

        apple = comp.get_vector(['apple'])
        the_apple = comp.get_vector(['the', 'apple'])
        quick_apple = comp.get_vector(['quick', 'apple'])
        the_quick_apple = comp.get_vector(['the', 'quick', 'apple'])
        the_fast_apple = comp.get_vector(['the', 'fast', 'apple'])

        assert cosine(apple, the_apple) < cosine(apple, quick_apple)
        assert cosine(apple, quick_apple) < cosine(apple, the_quick_apple)
        assert cosine(apple, the_fast_apple) < cosine(apple, the_quick_apple)
