import os
import numpy as np
import logging
import os
from .iterators import FileIterator, DirIterator, DirIterator, FileLineIterator, \
    TokenizedSequenceIterator, TokenIterator, SlidingWindowIterator
from .tokenization import DEFAULT_TOKENIZER, DEFAULT_SENT_TOKENIZER, DEFAULT_JAP_TOKENIZER
from vecto.utils.metadata import WithMetaData
from vecto.utils.data import detect_archive_format_and_open
logger = logging.getLogger(__name__)


class BaseCorpus(WithMetaData):
    """Cepresents a body of text in single or multiple files"""

    def __init__(self, path, language='eng'):
        self.path = path
        self.language = language

    def get_sliding_window_iterator(self, left_ctx_size=2, right_ctx_size=2, tokenizer=None, verbose=0):
        if tokenizer is None:
            if self.language == 'jap':
                tokenizer = DEFAULT_JAP_TOKENIZER
            else:
                tokenizer = DEFAULT_TOKENIZER
        return SlidingWindowIterator(
            self.get_sentence_iterator(tokenizer=tokenizer),
            left_ctx_size=left_ctx_size,
            right_ctx_size=right_ctx_size)

    def get_token_iterator(self, tokenizer=None, verbose=False):
        if tokenizer is None:
            if self.language == 'jap':
                tokenizer = DEFAULT_JAP_TOKENIZER
            else:
                tokenizer = DEFAULT_TOKENIZER
        return TokenIterator(self.get_sentence_iterator(tokenizer, verbose))

    def get_sentence_iterator(self, tokenizer=None, verbose=False):
        if tokenizer is None:
            if self.language == 'jap':
                tokenizer = DEFAULT_JAP_TOKENIZER
            else:
                tokenizer = DEFAULT_SENT_TOKENIZER
        return TokenizedSequenceIterator(self.get_line_iterator(verbose=verbose), tokenizer=tokenizer, verbose=verbose)


#class Corpus(BaseCorpus) #think of beter naming/renaming
    # def init() 

 #self.metadata is here

    # either master does plits and send each worker each split
    # or first send whole thing, and each worker does split
    # def get_view(start, end)
        #  return viuew

class SegmentIterator():
    def __init__(self, tree):
        # iterate from given file and offste
        pass


def get_uncompressed_size(path):
    with detect_archive_format_and_open(path) as f:
        size = f.seek(0, 2)
    return size


class ViewCorpus(BaseCorpus):
    # is returned from get_view from Corpus
    def load_dir_strucute(self):
        self.tree = []
        self.accumulated_size = 0
        for file in DirIterator(self.path):
            self.accumulated_size += get_uncompressed_size(file)
            self.tree.append((file, self.accumulated_size))
        print(self.tree)

    def get_line_iterator(self):
        # iterate over precomputed tree of files and sizes
        # iterated this file/this offset to last-file last offset
        pass


class FileCorpus(BaseCorpus):
    """Cepresents a body of text in a single file"""

    def get_line_iterator(self, verbose=False):
        return FileLineIterator(FileIterator(self.path, verbose=verbose))


class DirCorpus(BaseCorpus):
    """Cepresents a body of text in a directory"""

    def get_line_iterator(self, verbose=False):
        return FileLineIterator(DirIterator(self.path, verbose=verbose))


# old code below ----------------------------------


# def FileSlidingWindowCorpus(path, left_ctx_size=2, right_ctx_size=2, tokenizer=DEFAULT_TOKENIZER, verbose=0):
#    """
#    Reads text from `path` line-by-line, splits each line into tokens and/or sentences (depending on tokenizer),
#    and yields training samples for prediction-based distributional semantic models (like Word2Vec etc).
#    Example of one yielded value: {'current': 'long', 'context': ['family', 'dashwood', 'settled', 'sussex']}
#    :param path: text file to read (can be archived)
#    :param tokenizer: tokenizer to use to split into sentences and tokens
#    :param verbose: whether to enable progressbar or not
#    :return:
#    """
#    return SlidingWindowIterator(
#        TokenizedSequenceIterator(
#            FileLineIterator(
#                FileIterator(path, verbose=verbose)),
#            tokenizer=tokenizer),
#        left_ctx_size=left_ctx_size,
#        right_ctx_size=right_ctx_size)


def DirSlidingWindowCorpus(path, left_ctx_size=2, right_ctx_size=2, tokenizer=DEFAULT_TOKENIZER, verbose=0):
    """
    Reads text from all files from all subfolders of `path` line-by-line,
    splits each line into tokens and/or sentences (depending on `tokenizer`),
    and yields training samples for prediction-based distributional semantic models (like Word2Vec etc).
    Example of one yielded value: {'current': 'long', 'context': ['family', 'dashwood', 'settled', 'sussex']}
    :param path: text file to read (can be archived)
    :param tokenizer: tokenizer to use to split into sentences and tokens
    :param verbose: whether to enable progressbar or not
    :return:
    """
    return SlidingWindowIterator(
        TokenizedSequenceIterator(
            FileLineIterator(
                DirIterator(path, verbose=verbose)),
            tokenizer=tokenizer),
        left_ctx_size=left_ctx_size,
        right_ctx_size=right_ctx_size)


def corpus_chain(*corpuses):
    """
    Join all copuses into a big single one. Like `itertools.chain`, but with proper metadata handling.
    :param corpuses: other corpuses or iterators
    :return:
    """
    return IteratorChain(corpuses)


# TODO: make it a part of Corpus class
def load_path_as_ids(path, vocabulary, tokenizer=DEFAULT_TOKENIZER):
    # use proper tokenizer from cooc
    # options to ignore sentence bounbdaries
    # specify what to do with missing words
    # replace numbers with special tokens
    result = []
    if os.path.isfile(path):
        # TODO: why file corpus does not need language? 
        ti = FileCorpus(path).get_token_iterator(tokenizer=tokenizer)
    else:
        if os.path.isdir(path):
            ti = DirCorpus(path).get_token_iterator(tokenizer)
        else:
            raise RuntimeError("source file does not exist")
    for token in ti:
        w = token  # specify what to do with missing words
        result.append(vocabulary.get_id(w))
    return np.array(result, dtype=np.int32)
