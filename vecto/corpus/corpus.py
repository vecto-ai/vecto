import numpy as np
import logging
from .iterators import FileIterator, DirIterator, DirIterator, FileLineIterator, \
    TokenizedSequenceIterator, TokenIterator, IteratorChain, \
    SlidingWindowIterator
from .tokenization import DEFAULT_TOKENIZER, DEFAULT_SENT_TOKENIZER
from vecto.utils.metadata import WithMetaData


logger = logging.getLogger(__name__)


class Corpus(WithMetaData):
    """Cepresents a body of text in single or multiple files"""

    def __init__(self, path):
        self.path = path

    def get_token_iterator(self, tokenizer, verbose=False):
        return TokenIterator(self.get_sentence_iterator(tokenizer, verbose))

    def get_sentence_iterator(self, tokenizer, verbose=False):
        return TokenizedSequenceIterator(self.get_line_iterator(), tokenizer=tokenizer)


class FileCorpus(Corpus):
    """Cepresents a body of text in a single file"""

    def __init__(self, path):
        super.__init__()

    def get_line_iterator(self, tokenizer, verbose=False):
        return FileLineIterator(FileIterator(self.path, verbose=verbose))


class DirCorpus(Corpus):
    """Cepresents a body of text in a directory"""

    def __init__(self, path):
        super.__init__()

    def get_token_iterator(self, tokenizer, verbose=False):
        return TokenIterator(
            TokenizedSequenceIterator(
                FileLineIterator(
                    DirIterator(self.path, verbose=verbose)),
                tokenizer=tokenizer))

## old code below ----------------------------------

def FileSentenceCorpus(path, tokenizer=DEFAULT_SENT_TOKENIZER, verbose=0):
    """
    Reads text from `path` line-by-line, splits each line into sentences, tokenizes each sentence.
    Yields data sentence-by-sentence.
    :param path: text file to read (can be archived)
    :param tokenizer: tokenizer to use to split into sentences and tokens
    :param verbose: whether to enable progressbar or not
    :return:
    """
    return TokenizedSequenceIterator(
        FileLineIterator(
            FileIterator(path, verbose=verbose)),
        tokenizer=tokenizer)


def DirSentenceCorpus(path, tokenizer=DEFAULT_SENT_TOKENIZER, verbose=0):
    """
    Reads text from all files from all subfolders of `path` line-by-line,
    splits each line into sentences, tokenizes each sentence.
    Yields data sentence-by-sentence.
    :param path: root directory with text files
    :param tokenizer: tokenizer to use to split into sentences and tokens
    :param verbose: whether to enable progressbar or not
    :return:
    """
    return TokenizedSequenceIterator(
        FileLineIterator(
            DirIterator(path, verbose=verbose)),
        tokenizer=tokenizer)


def FileTokenCorpus(path, tokenizer=DEFAULT_TOKENIZER, verbose=0):
    """
    Reads text from `path` line-by-line, splits each line into tokens.
    Yields data token-by-token.
    :param path: text file to read (can be archived)
    :param tokenizer: tokenizer to use to split into sentences and tokens
    :param verbose: whether to enable progressbar or not
    :return:
    """
    return TokenIterator(
        TokenizedSequenceIterator(
            FileLineIterator(
                FileIterator(path, verbose=verbose)),
            tokenizer=tokenizer))


def DirTokenCorpus(path, tokenizer=DEFAULT_TOKENIZER, verbose=0):
    """
    Reads text from all files from all subfolders of `path` line-by-line, splits each line into tokens.
    Yields data token-by-token.
    :param path: text file to read (can be archived)
    :param tokenizer: tokenizer to use to split into sentences and tokens
    :param verbose: whether to enable progressbar or not
    :return:
    """
    return TokenIterator(
        TokenizedSequenceIterator(
            FileLineIterator(
                DirIterator(path, verbose=verbose)),
            tokenizer=tokenizer))


def FileSlidingWindowCorpus(path, left_ctx_size=2, right_ctx_size=2, tokenizer=DEFAULT_TOKENIZER, verbose=0):
    """
    Reads text from `path` line-by-line, splits each line into tokens and/or sentences (depending on tokenizer),
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
                FileIterator(path, verbose=verbose)),
            tokenizer=tokenizer),
        left_ctx_size=left_ctx_size,
        right_ctx_size=right_ctx_size)


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


def load_file_as_ids(path, vocabulary, tokenizer=DEFAULT_TOKENIZER):
    # use proper tokenizer from cooc
    # options to ignore sentence bounbdaries
    # specify what to do with missing words
    # replace numbers with special tokens
    result = []
    ti = FileTokenCorpus(path, tokenizer=tokenizer)
    for token in ti:
        w = token    # specify what to do with missing words
        result.append(vocabulary.get_id(w))
    return np.array(result, dtype=np.int32)
