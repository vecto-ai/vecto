import numpy as np
import logging
from .iterators import FileIterator, DirIterator, DirIterator, FileLineIterator, EntireFileIterator, \
    TokenizedSequenceIterator, TokenIterator, SlidingWindowIterator
from .tokenization import DEFAULT_TOKENIZER, DEFAULT_SENT_TOKENIZER
from vecto.utils.metadata import WithMetaData


logger = logging.getLogger(__name__)


class Corpus(WithMetaData):
    """Represents a body of text in single or multiple files"""

    def __init__(self, path):
        super(Corpus, self).__init__()
        self.path = path

    def get_sliding_window_iterator(self, left_ctx_size=2, right_ctx_size=2, tokenizer=DEFAULT_TOKENIZER, verbose=0):
        return SlidingWindowIterator(
            self.get_sentence_iterator(tokenizer=tokenizer),
            left_ctx_size=left_ctx_size,
            right_ctx_size=right_ctx_size)

    def get_token_iterator(self, tokenizer=DEFAULT_TOKENIZER, verbose=False):
        return TokenIterator(self.get_sentence_iterator(tokenizer, verbose))

    def get_sentence_iterator(self, tokenizer=DEFAULT_SENT_TOKENIZER, verbose=False):
        return TokenizedSequenceIterator(self.get_text_iterator(), tokenizer=tokenizer, verbose=verbose)


class FileCorpus(Corpus):
    """Represents a body of text in a single file"""

    def get_text_iterator(self, verbose=False):
        return FileLineIterator(FileIterator(self.path, verbose=verbose))


class DirCorpus(Corpus):
    """Represents a body of text in a directory"""

    def __init__(self, path, by_line=True):
        super(DirCorpus, self).__init__(path)
        self.metadata['by_line'] = by_line
        self.by_line = by_line

    def get_text_iterator(self, verbose=False):
        base = DirIterator(self.path, verbose=verbose)
        if self.by_line:
            return FileLineIterator(base)
        else:
            return EntireFileIterator(base)


# old code below ----------------------------------


#def FileSlidingWindowCorpus(path, left_ctx_size=2, right_ctx_size=2, tokenizer=DEFAULT_TOKENIZER, verbose=0):
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
def load_file_as_ids(path, vocabulary, tokenizer=DEFAULT_TOKENIZER):
    # use proper tokenizer from cooc
    # options to ignore sentence bounbdaries
    # specify what to do with missing words
    # replace numbers with special tokens
    result = []
    ti = FileCorpus(path).get_token_iterator(tokenizer=tokenizer)
    for token in ti:
        w = token    # specify what to do with missing words
        result.append(vocabulary.get_id(w))
    return np.array(result, dtype=np.int32)
