import collections
import fnmatch
import logging
import os

from vecto.corpus.base import BaseIterator
from vecto.corpus.tokenization import DEFAULT_SENT_TOKENIZER, DEFAULT_TOKENIZER
from vecto.utils.data import detect_archive_format_and_open

logger = logging.getLogger(__name__)


class FileIterator(BaseIterator):
    """
    Iterator which yields only given filename.
    """

    def __init__(self, filename, verbose=0):
        super(FileIterator, self).__init__(base_path=filename,
                                           verbose=verbose)
        self.filename = filename

    def _generate_samples(self):
        yield self.filename


class DirIterator(BaseIterator):
    """
    Iterator which yield all files in the given folder and all its subfolders.
    """

    def __init__(self, dirname, verbose=0):
        super(DirIterator, self).__init__(base_path=dirname,
                                          verbose=verbose)
        self.dirname = dirname

    def _generate_samples(self):
        for root, _, files in os.walk(self.dirname, followlinks=True):
            for good_fname in sorted(fnmatch.filter(files, "*")):
                full_file_path = os.path.join(root, good_fname)
                logger.info("processing " + full_file_path)
                yield full_file_path


class FileLineIterator(BaseIterator):
    """
    Receives a sequence of filenames from `base_corpus` and reads each file line-by-line.
    """

    def __init__(self, base_corpus, verbose=0):
        super(FileLineIterator, self).__init__(base_corpus=base_corpus.metadata,
                                               verbose=verbose)
        self.base_corpus = base_corpus

    def _generate_samples(self):
        for filename in self.base_corpus:
            with detect_archive_format_and_open(filename) as file_in:
                for line in file_in:
                    line = line.strip()
                    if line:
                        yield line


def seek_unicode(fp, position, direction=-1):
    while position >= 0:
        fp.seek(position)
        try:
            fp.seek(position)
            fp.read(1)
            fp.seek(position)
            return
        except UnicodeDecodeError:
            position += direction
    raise UnicodeDecodeError("File not decodable")


class ViewLineIterator(BaseIterator):
    def __init__(self, tree, start, end, verbose):
        # TODO: sort this stuff from parent class out
        super().__init__(base_corpus=None, verbose=verbose)
        self.tree = tree
        self.start = start
        self.end = end

    def _generate_samples(self):
        for i in range(self.start[0], self.end[0] + 1):
            filename = self.tree[i].filename
            with detect_archive_format_and_open(filename) as file_in:
                if i == self.start[0]:
                    # TODO: conside seek to beginning of line
                    seek_unicode(file_in, self.start[1])
                cnt_bytes_read = self.start[0] if self.start[0] == self.end[0] else 0
                for line in file_in:
                    cnt_bytes_read += len(line)
                    line = line.strip()
                    if cnt_bytes_read > self.end[1]:
                        break
                    yield line


class LoopedLineIterator(BaseIterator):
    def __init__(self, tree, start):
        super().__init__()
        self.tree = tree
        self.id_file = start[0]
        self.start_offset = start[1]
        self._cnt_restarts = 0

    def _generate_samples(self):
        filename = self.tree[self.id_file][0]
        file_in = detect_archive_format_and_open(filename)
        seek_unicode(file_in, self.start_offset)
        while True:
            for line in file_in:
                line = line.strip()
                yield line
            file_in.close()
            self.id_file += 1
            if self.id_file >= len(self.tree):
                self.id_file = 0
                self._cnt_restarts += 1
            file_in = detect_archive_format_and_open(self.tree[self.id_file][0])

    @property
    def cnt_restarts(self):
        return self._cnt_restarts


class TokenizedSequenceIterator(BaseIterator):
    """
    Receives any corpus yielding text (e.g. `FileLineIterator`) and produces tokenized sequences.
    Good for splitting texts on sentences.
    """

    def __init__(self, base_corpus, tokenizer=DEFAULT_TOKENIZER, verbose=0):
        super(TokenizedSequenceIterator, self).__init__(base_corpus=base_corpus.metadata,
                                                        tokenizer=tokenizer.metadata,
                                                        verbose=verbose)
        self.base_corpus = base_corpus
        self.tokenizer = tokenizer

    def _generate_samples(self):
        for line in self.base_corpus:
            # TODO: sentence may span over multiple lines, we should take this into account somehow
            # I think that it's better to ignore this here and write docs like:
            # "You should be aware of that and prepare your data accordingly, e.g. one line - one real doc"
            tokenized = self.tokenizer(line.strip())
            for tokenized_sentence in tokenized:
                yield tokenized_sentence


class SequenceIterator(BaseIterator):
    def __init__(self, line_terator, sequence_length, tokenizer, minimal_length=0, reset_on_new_line=False):
        super().__init__()
        self.line_iterator = line_terator
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.buffer = []
        self.minimal_length = minimal_length
        self.reset_on_new_line = reset_on_new_line

    def _generate_samples(self):
        # TODO: consider removing too small chunks of sentences at the end
        # TODO: consider leveraging sentence iterator is corpus has mark-up
        for line in self.line_iterator:
            tokens = self.tokenizer(line)
            if self.reset_on_new_line:
                self.buffer = []
            elif len(self.buffer) < self.minimal_length:
                self.buffer = []
            self.buffer += tokens
            while len(self.buffer) > self.sequence_length - self.minimal_length:
                s = self.buffer[: self.sequence_length]
                self.buffer = self.buffer[self.sequence_length:]
                yield s

    @property
    def cnt_restarts(self):
        # TODO: this will fail with non-looped line iterator,
        # maybe there's a way to do it more gracefully
        return self.line_iterator.cnt_restarts


class BaseNestedIterator(BaseIterator):
    def __init__(self, parent_iterator, verbose=0):
        # TODO: this .metadata seems strange
        super().__init__(parent_iterator=parent_iterator.metadata,
                         verbose=verbose)
        self.parent_iterator = parent_iterator


class TokenIterator(BaseNestedIterator):

    def _generate_samples(self):
        for tokenized_str in self.parent_iterator:
            for token in tokenized_str:
                yield token


def iter_sliding_window(seq, left_ctx_size, right_ctx_size):
    for i, current in enumerate(seq):
        ctx = []
        ctx.extend(seq[i - left_ctx_size: i])
        ctx.extend(seq[i + 1: i + right_ctx_size + 1])
        yield i, current, ctx


class SlidingWindowIterator(BaseIterator):
    """
    Receives any corpus yielding sequences of tokens (e.g. TokenizedSequenceIterator)
    and produces training samples for prediction-based distributional semantic models (like Word2Vec etc).
    Example of one yielded value: {'current': 'long', 'context': ['family', 'dashwood', 'settled', 'sussex']}
    """

    def __init__(self, base_corpus, left_ctx_size=2, right_ctx_size=2, verbose=0):
        assert isinstance(next(iter(base_corpus)), collections.abc.Sequence)
        super(SlidingWindowIterator, self).__init__(base_corpus=base_corpus.metadata,
                                                    left_ctx_size=left_ctx_size,
                                                    right_ctx_size=right_ctx_size,
                                                    verbose=verbose)
        self.base_corpus = base_corpus
        self.left_ctx_size = left_ctx_size
        self.right_ctx_size = right_ctx_size
        self.__gen__  = self._generate_samples()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.__gen__)

    def _generate_samples(self):
        for sample_elems in self.base_corpus:
            for _, current, ctx in iter_sliding_window(sample_elems,
                                                       self.left_ctx_size,
                                                       self.right_ctx_size):
                yield dict(current=current,
                           context=ctx)


# class SlidingWindowAndGlobal(BaseIterator):
#     def __init__(self, base_corpus, left_ctx_size=2, right_ctx_size=2, verbose=0):
#         assert isinstance(next(iter(base_corpus)), collections.abc.Sequence)
#         super(SlidingWindowAndGlobal, self).__init__(base_corpus=base_corpus.metadata,
#                                                      left_ctx_size=left_ctx_size,
#                                                      right_ctx_size=right_ctx_size,
#                                                      verbose=verbose)
#         self.base_corpus = base_corpus
#         self.left_ctx_size = left_ctx_size
#         self.right_ctx_size = right_ctx_size

#     def _generate_samples(self):
#         for sample_elems in self.base_corpus:
#             for _, current, ctx in iter_sliding_window(sample_elems,
#                                                        self.left_ctx_size,
#                                                        self.right_ctx_size):
#                 yield dict(current=current,
#                            context=ctx,
#                            global_context=list(sample_elems))


# class IteratorChain(BaseIterator):
#    """
#    Like `itertools.chain`, but with proper metadata handling
#    """
#    def __init__(self, base_iterators, verbose=0):
#        super(IteratorChain, self).__init__(base_iterators=[i.metadata for i in base_iterators],
#                                            verbose=verbose)
#        self.base_iterators = base_iterators

#    def _generate_samples(self):
#        for base_iter in self.base_iterators:
#            for sample in base_iter:
#                yield sample


# class TruncatedCorpus(BaseIterator):
#    """
#    Reads first `limit` samples from `base_corpus` and yields them sample-by-sample.
#    Good for debugging.
#    """
#    def __init__(self, base_corpus, limit=1000, verbose=0):
#        super(TruncatedCorpus, self).__init__(base_corpus=base_corpus.meta,
#                                              verbose=verbose)
#        self.samples = []
#        for i, s in enumerate(base_corpus):
#            if i >= limit:
#                break
#            self.samples.append(s)
#        self.metadata['samples_count'] = len(self.samples)

#    def _generate_samples(self):
#        for s in self.samples:
#            yield s
