"""Tests for embeddings module."""

import unittest
from unittest.mock import patch
from os import path
import numpy as np
from vecto.embeddings.dense import WordEmbeddingsDense
from vecto.embeddings.base import WordEmbeddings
from vecto.embeddings import load_from_dir
from vecto.vocabulary import Vocabulary


class Tests(unittest.TestCase):

    def test_basic(self):
        WordEmbeddingsDense()
        model = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        model.cmp_words("apple", "banana")
        model.cmp_words("apple", "bananaaaaa")
        x = np.array([0.0, 0.0, 0.0])
        x.fill(np.nan)
        model.cmp_vectors(x, x)

    def test_load(self):
        load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        # TODO: assert right class
        load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_no_file_header'))
        # TODO: assert right class
        load_from_dir(path.join('tests', 'data', 'embeddings', 'npy'))

        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        embs.get_vector('apple')
        #with self.assertRaises(RuntimeError):
        #    embs.get_vector('word_that_not_in_vocabulary_27')
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'corrupted'))
        with self.assertRaises(RuntimeError):
            embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text'))

    def test_normalize(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        embs.normalize()
        embs.cache_normalized_copy()

    def test_utils(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        results = embs.get_most_similar_words('apple', 5)
        print(results)
        embs.cache_normalized_copy()
        results = embs.get_most_similar_words('apple', 5)
        print(results)

        results = embs.get_most_similar_words(embs.get_vector('apple'), 5)
        print(results)
        embs.get_x_label(0)

    def test_save(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        path_save = path.join('/tmp', 'vecto', 'saved')
        embs.save_to_dir(path_save)
        embs = load_from_dir(path_save)
        print(embs.matrix.shape)
        embs.save_to_dir_plain_txt(path.join('/tmp', 'vecto', 'saved_plain'))

    def test_filter(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        path_vocab = path.join('.', 'tests', 'data', 'vocabs', 'plain')
        vocab = Vocabulary()
        vocab.load(path_vocab)
        embs.filter_by_vocab(["the", "apple"])
        embs.filter_by_vocab([])

    @patch.multiple(WordEmbeddings, __abstractmethods__=set())
    def test_abc(self):
        obj = WordEmbeddings()
        obj.get_vector("banana")

    def test_viz(self):
        embs = load_from_dir(path.join('tests', 'data', 'embeddings', 'text', 'plain_with_file_header'))
        embs.viz_wordlist(["the", "apple"], colored=True, show_legend=True)
        embs.viz_wordlist(["the", "apple"], colored=False, show_legend=False)
