"""Tests for corpus module."""

import unittest
import numpy as np
import json
from vecto.corpus import FileCorpus, DirCorpus, corpus_chain, load_file_as_ids
from vecto.vocabulary import Vocabulary

# todo: use local vocab
path_vocab = "./tests/data/vocabs/plain"
path_text = "./tests/data/corpora/plain"
path_gzipped = "./tests/data/corpora/gzipped"
path_bzipped = "./tests/data/corpora/bzipped"
path_text_file = "./tests/data/corpora/plain/sense_small.txt"


def count_words_and_collect_prefix(corpus, max_len=10):
    total_words = 0
    words = []
    for w in corpus:
        if len(words) < max_len:
            words.append(w)
        total_words += 1
    return total_words, words


TEST_TEXT_LEN = 4207
TEST_FIRST_10_WORDS = 'family|dashwood|long|settled|sussex|estate|large|residence|norland|park'


TEST_RIGHT_METADATA = r'''
{
    "_class": "vecto.corpus.iterators.IteratorChain",
    "base_iterators": [
        {
            "_class": "vecto.corpus.iterators.TokenIterator",
            "base_corpus": {
                "_class": "vecto.corpus.iterators.TokenizedSequenceIterator",
                "base_corpus": {
                    "_class": "vecto.corpus.iterators.FileLineIterator",
                    "base_corpus": {
                        "_base_path": "./tests/data/corpora/plain/sense_small.txt",
                        "_class": "vecto.corpus.iterators.FileIterator",
                        "vecto_version": "0.1.1"
                    },
                    "vecto_version": "0.1.1"
                },
                "tokenizer": {
                    "_class": "vecto.corpus.tokenization.Tokenizer",
                    "good_token_re": "^\\w+$",
                    "min_token_len": 3,
                    "normalizer": "vecto.corpus.tokenization.default_token_normalizer",
                    "stopwords": "too long to be saved to metadata, i suppose",
                    "vecto_version": "0.1.1"
                },
                "vecto_version": "0.1.1"
            },
            "vecto_version": "0.1.1"
        },
        {
            "_class": "vecto.corpus.iterators.TokenIterator",
            "base_corpus": {
                "_class": "vecto.corpus.iterators.TokenizedSequenceIterator",
                "base_corpus": {
                    "_class": "vecto.corpus.iterators.FileLineIterator",
                    "base_corpus": {
                        "_base_path": "./tests/data/corpora/plain",
                        "_class": "vecto.corpus.iterators.DirIterator",
                        "vecto_version": "0.1.1"
                    },
                    "vecto_version": "0.1.1"
                },
                "tokenizer": {
                    "_class": "vecto.corpus.tokenization.Tokenizer",
                    "good_token_re": "^\\w+$",
                    "min_token_len": 3,
                    "normalizer": "vecto.corpus.tokenization.default_token_normalizer",
                    "stopwords": "too long to be saved to metadata, i suppose",
                    "vecto_version": "0.1.1"
                },
                "vecto_version": "0.1.1"
            },
            "vecto_version": "0.1.1"
        }
    ],
    "vecto_version": "0.1.1"
}
'''.strip()


class Tests(unittest.TestCase):

    def test_file_corpus(self):
        corpus = FileCorpus(path_text_file)
        tokens_iter = corpus.get_token_iterator(verbose=1)
        total_words, words = count_words_and_collect_prefix(tokens_iter)
        print("!!!!!total words", total_words)
        assert total_words == TEST_TEXT_LEN
        assert '|'.join(words) == TEST_FIRST_10_WORDS

    def test_dir_corpus(self):
        corpus = DirCorpus(path_text)
        tokens_iter = corpus.get_token_iterator()
        total_words, words = count_words_and_collect_prefix(tokens_iter)
        assert total_words == TEST_TEXT_LEN
        assert '|'.join(words) == TEST_FIRST_10_WORDS

    def test_dir_iter_gzipped(self):
        corpus = DirCorpus(path_gzipped)
        tokens_iter = corpus.get_token_iterator()
        total_words, words = count_words_and_collect_prefix(tokens_iter)
        assert total_words == TEST_TEXT_LEN
        assert '|'.join(words) == TEST_FIRST_10_WORDS

    def test_dir_iter_bzipped(self):
        corpus = DirCorpus(path_bzipped)
        tokens_iter = corpus.get_token_iterator()
        total_words, words = count_words_and_collect_prefix(tokens_iter)
        assert total_words == TEST_TEXT_LEN
        assert '|'.join(words) == TEST_FIRST_10_WORDS

    def test_sentence(self):
        corpus = FileCorpus(path_text_file)
        sentence_iter = corpus.get_sentence_iterator(verbose=True)
        for s in sentence_iter:
            assert s == ['family', 'dashwood', 'long', 'settled', 'sussex']
            break

    def test_sliding_window(self):
        corpus = FileCorpus(path_text_file)
        sliding_window_iter = corpus.get_sliding_window_iterator()
        for i, s in enumerate(sliding_window_iter):
            if i >= 2:
                break
        assert s == {'current': 'long', 'context': ['family', 'dashwood', 'settled', 'sussex']}

# ----old tests ---------------------

    def test_text_to_ids(self):
        v = Vocabulary()
        v.load(path_vocab)
        doc = load_file_as_ids(path_text_file, v)
        assert doc.shape == (TEST_TEXT_LEN,)
        assert np.allclose(doc[:10], [-1, 40, -1, -1, -1, -1, -1, -1, 57, -1])


    #def test_chain(self):
    #    total_words, words = count_words_and_collect_prefix(corpus_chain(FileTokenCorpus(path_text_file),
    #                                                                     DirTokenCorpus(path_text)))
    #    assert total_words == TEST_TEXT_LEN * 2
    #    assert '|'.join(words) == TEST_FIRST_10_WORDS

    #def test_metadata(self):
    #    corp = corpus_chain(FileTokenCorpus(path_text_file),
    #                        DirTokenCorpus(path_text))
    #    metadata = json.dumps(corp.metadata, indent=4, sort_keys=True).strip()
    #    assert metadata == TEST_RIGHT_METADATA


