"""Tests for corpus module."""

import unittest
from vecto.corpus import load_file_as_ids, FileTokenIterator, DirTokenIterator
from vecto.vocabulary import Vocabulary

# todo: use local vocab
path_vocab = "./tests/data/vocabs/plain"
path_text = "./tests/data/corpora/plain"
path_gzipped = "./tests/data/corpora/gzipped"
path_bzipped = "./tests/data/corpora/bzipped"
path_text_file = "./tests/data/corpora/plain/sense_small.txt"


class Tests(unittest.TestCase):

    def test_file_iter(self):
        cnt = 0
        print()
        for w in (FileTokenIterator(path_text_file)):
            if cnt < 16:
                print(w, end=" | ")
            cnt += 1
        print()
        print(cnt, "words read")

    def test_dir_iter(self):
        cnt = 0
        it = DirTokenIterator(path_text)
        for w in it:
            cnt += 1
        print(cnt, "words read")

    def test_text_to_ids(self):
        v = Vocabulary()
        v.load(path_vocab)
        doc = load_file_as_ids(path_text_file, v, downcase=False)
        doc = load_file_as_ids(path_text_file, v)
        print("test load as ids:", doc.shape)
        print(doc[:10])

    def test_dir_iter_gzipped(self):
        cnt = 0
        for w in (DirTokenIterator(path_gzipped)):
            cnt += 1
        print(cnt, "words read")

    def test_dir_iter_bzipped(self):
        cnt = 0
        for w in (DirTokenIterator(path_bzipped)):
            cnt += 1
        print(cnt, "words read")
