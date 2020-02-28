"""Tests for embeddings module."""

import unittest
import io
import contextlib
import sys
import runpy
import os


def run_module(name: str, args, run_name: str = '__main__') -> None:
    backup_sys_argv = sys.argv
    sys.argv = [name + '.py'] + list(args)
    runpy.run_module(name, run_name=run_name)
    sys.argv = backup_sys_argv


class Tests(unittest.TestCase):

    # def test_train_word2vec(self):
    #     path_corpus = "./tests/data/corpora/plain/"
    #     sio = io.StringIO()
    #     with contextlib.redirect_stderr(sio):
    #         run_module('vecto.embeddings.train_word2vec',
    #                    ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--out_type', 'ns'])
    #         run_module('vecto.embeddings.train_word2vec',
    #                    ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--out_type', 'hsm'])
    #         run_module('vecto.embeddings.train_word2vec',
    #                    ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--out_type', 'original'])
    #         run_module('vecto.embeddings.train_word2vec',
    #                    ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--out_type', 'ns',
    #                     '--model', 'cbow'])
    #         with self.assertRaises(RuntimeError):
    #             run_module('vecto.embeddings.train_word2vec',
    #                        ['--path_corpus', path_corpus + "NONEXISTING", '--path_out', '/tmp/vecto/embeddings/',
    #                         '--out_type', 'ns',
    #                         '--model', 'cbow'])

    # @unittest.skipIf(os.environ.get('APPVEYOR'), 'skipping Appveyor due to memory error')
    # def test_train_word2vec_subword_cnn1d(self):
    #     path_corpus = "./tests/data/corpora/plain/"
    #     run_module('vecto.embeddings.train_word2vec',
    #                ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
    #                 '--subword', 'cnn1d'])
    #     with self.assertRaises(RuntimeError):
    #         run_module('vecto.embeddings.train_word2vec',
    #                    ['--path_corpus', path_corpus + "NONEXISTING", '--path_out', '/tmp/vecto/embeddings/',
    #                     '--dimension', '5',
    #                     '--subword', 'cnn1d'])

    def test_train_word2vec_subword(self):
        path_corpus = "./tests/data/corpora/plain/"
        path_vocab = "./tests/data/vocabs/plain/"
        sio = io.StringIO()
        with contextlib.redirect_stderr(sio):
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
                        '--subword', 'cnn1d_small'])

            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
                        '--subword', 'bilstm'])
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
                        '--subword', 'sum'])
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
                        '--subword', '_none', '--path_vocab', path_vocab])
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
                        '--subword', 'bilstm_sum'])
            with self.assertRaises(RuntimeError):
                run_module('vecto.embeddings.train_word2vec',
                           ['--path_corpus', path_corpus + "NONEXISTING", '--path_out', '/tmp/vecto/embeddings/',
                            '--dimension', '5',
                            '--subword', 'bilstm_sum'])

    def test_train_word2vec_subword_jap(self):
        path_corpus = "./tests/data/corpora/jap/tokenized/"
        path_word2chars = "./tests/data/corpora/jap/char2radical/char2radical.txt"
        sio = io.StringIO()
        with contextlib.redirect_stderr(sio):
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
                        '--subword', 'sum', '--language', 'jap', '--min_gram', '1', '--max_gram', '1'])
            run_module('vecto.embeddings.train_word2vec',
                       ['--path_corpus', path_corpus, '--path_out', '/tmp/vecto/embeddings/', '--dimension', '5',
                        '--subword', 'sum', '--language', 'jap', '--min_gram', '1', '--max_gram', '1',
                        '--path_word2chars', path_word2chars])

            with self.assertRaises(RuntimeError):
                run_module('vecto.embeddings.train_word2vec',
                           ['--path_corpus', path_corpus + "NONEXISTING", '--path_out', '/tmp/vecto/embeddings/',
                            '--dimension', '5',
                            '--subword', 'sum', '--language', 'jap', '--min_gram', '1', '--max_gram', '1'])
