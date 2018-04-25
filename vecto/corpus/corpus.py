import numpy as np
import fnmatch
import os
import re
from vecto.utils.data import detect_archive_format_and_open
import logging

logger = logging.getLogger(__name__)

_default_tokenizer_patter = r"[\w\-']+|[.,!?…]"


class LineTokenizer:

    def __init__(self, re_pattern=_default_tokenizer_patter):
        self.re_token = re.compile(re_pattern)

    def __call__(self, s):
        tokens = self.re_token.findall(s)
        return tokens


class FileTokenIterator:

    def __init__(self, path, re_pattern=_default_tokenizer_patter):
        self.path = path
        self.tokenizer = LineTokenizer(re_pattern)

    def __iter__(self):
        return self.next()

    def next(self):
        with detect_archive_format_and_open(self.path) as f:
            for line in f:
                s = line.strip().lower()
                # todo lower should be parameter
                tokens = self.tokenizer(s)
                for token in tokens:
                    yield token


class DirTokenIterator:
    def __init__(self, path, re_pattern=_default_tokenizer_patter):
        self.path = path
        self.__gen__ = self.gen(re_pattern)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.__gen__)

    def gen(self, re_pattern):
        for root, dir, files in os.walk(self.path, followlinks=True):
            for items in fnmatch.filter(files, "*"):
                logger.info("processing " + os.path.join(root, items))
                for token in FileTokenIterator(os.path.join(root, items), re_pattern=re_pattern):
                    yield(token)


<<<<<<< HEAD
def load_file_as_ids(path, vocabulary, gzipped=None, downcase=True, re_pattern=_default_tokenizer_patter):
=======
def load_file_as_ids(path, vocabulary, gzipped=None, downcase=True, re_pattern=r"[\w\-']+|[.,!?…]"):
>>>>>>> 695d3cc... add corpus module
    # use proper tokenizer from cooc
    # options to ignore sentence bounbdaries
    # specify what to do with missing words
    # replace numbers with special tokens
    result = []
    ti = FileTokenIterator(path, re_pattern=re_pattern)
    for token in ti:
        w = token    # specify what to do with missing words
        if downcase:
            w = w.lower()
        result.append(vocabulary.get_id(w))
    return np.array(result, dtype=np.int32)
<<<<<<< HEAD
=======


def main():
    print("test")
>>>>>>> 695d3cc... add corpus module
