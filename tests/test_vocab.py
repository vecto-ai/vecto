"""Tests for vocabulary module."""

import unittest
import runpy
import io
import sys
import contextlib
from vecto.vocabulary import create_from_dir, create_from_file, Vocabulary

path_text = "./tests/data/corpora/plain"
path_text_file = "./tests/data/corpora/plain/sense_small.txt"
path_vocab = "./tests/data/vocabs/plain"


def run_module(name: str, *args, run_name: str = '__main__') -> None:
    backup_sys_argv = sys.argv
    sys.argv = [name + '.py'] + list(args)
    runpy.run_module(name, run_name=run_name)
    sys.argv = backup_sys_argv


class Tests(unittest.TestCase):

    def test_create_from_dir(self):
        vocab = create_from_dir(path_text, min_frequency=10)
        print("the:", vocab.get_id("the"))
        assert vocab.get_id("the") >= 0
        vocab.save_to_dir("/tmp/vecto/vocab")
        vocab = create_from_file(path_text_file, min_frequency=10)
        assert vocab.get_id("the") >= 0

    def test_load_from_dir(self):
        vocab = Vocabulary()
        vocab.load(path_vocab)
        print("the:", vocab.get_id("the"))

    def test_tokens_to_ids(self):
        vocab = Vocabulary()
        vocab.load(path_vocab)
        tokens = ["the", "apple"]
        ids = vocab.tokens_to_ids(tokens)
        print("ids:", ids)

    def test_misc(self):
        vocab = Vocabulary()
        vocab.load(path_vocab)
        vocab.get_word_by_id(1)
        vocab.get_frequency("apple")

    def test_cli(self):
        sio = io.StringIO()
        with contextlib.redirect_stderr(sio):
            with self.assertRaises(ValueError):
                run_module('version_query', '-p', '-i', '.')
        # _LOG.info('%s', sio.getvalue())
