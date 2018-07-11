"""Tests for embeddings module."""

import unittest
from unittest.mock import patch
import numpy as np
import io
import contextlib
import sys
import runpy
from vecto.embeddings.dense import WordEmbeddingsDense
from vecto.embeddings.base import WordEmbeddings
from vecto.embeddings import load_from_dir
from vecto.vocabulary import Vocabulary


def run_module(name: str, args, run_name: str = '__main__') -> None:
    backup_sys_argv = sys.argv
    sys.argv = [name + '.py'] + list(args)
    runpy.run_module(name, run_name=run_name)
    sys.argv = backup_sys_argv


class Tests(unittest.TestCase):

    def test_train_word2vec(self):
        path_corpus = "./tests/data/corpora/plain/"
        sio = io.StringIO()
        with contextlib.redirect_stderr(sio):
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--out_type', 'ns'])
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--out_type', 'hsm'])
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--out_type', 'original'])
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--out_type', 'ns',
                        '--model', 'cbow'])

    def test_train_word2vec_subword(self):
        path_corpus = "./tests/data/corpora/plain/"
        sio = io.StringIO()
        with contextlib.redirect_stderr(sio):
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
                        '--subword', 'cnn1d'])
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
                        '--subword', 'bilstm'])
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
                        '--subword', 'sum'])
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
                        '--subword', '_none'])
