import numpy as np
import collections
from .base import BaseCorpus
from .tokenization import DEFAULT_TOKENIZER
from vecto.utils.data import detect_archive_format_and_open


class FileTokenIterator(BaseCorpus):
    def __init__(self, base_corpus, tokenizer=DEFAULT_TOKENIZER, verbose=1):
        super(FileTokenIterator, self).__init__(base_corpus=base_corpus.metadata,
                                                tokenizer=str(tokenizer),
                                                verbose=verbose)
        self.base_corpus = base_corpus
        self.tokenizer = tokenizer

    def _generate_samples(self):
        for filename in self.base_corpus:
            with detect_archive_format_and_open(filename) as f:
                for line in f:
                    for token_lst in self.tokenizer(line.strip()):
                        for token in token_lst:
                            yield token


def iter_sliding_window(seq, left_ctx_size, right_ctx_size):
    for i, current in enumerate(seq):
        ctx = []
        ctx.extend(seq[i - left_ctx_size : i])
        ctx.extend(seq[i + 1 : i + right_ctx_size + 1])
        yield i, current, ctx


class SlidingWindowIterator(BaseCorpus):
    def __init__(self, base_corpus, left_ctx_size=2, right_ctx_size=2, verbose=1):
        assert isinstance(next(iter(base_corpus)), collections.abc.Sequence)
        super(SlidingWindowIterator, self).__init__(base_corpus=base_corpus.metadata,
                                                    left_ctx_size=left_ctx_size,
                                                    right_ctx_size=right_ctx_size,
                                                    verbose=verbose)
        self.base_corpus = base_corpus
        self.left_ctx_size = left_ctx_size
        self.right_ctx_size = right_ctx_size

    def _generate_samples(self):
        for sample_elems in self.base_corpus:
            for _, current, ctx in iter_sliding_window(sample_elems,
                                                       self.left_ctx_size,
                                                       self.right_ctx_size):
                yield dict(current=current,
                           context=ctx)


class SlidingWindowAndGlobal(BaseCorpus):
    def __init__(self, base_corpus, left_ctx_size=2, right_ctx_size=2, verbose=1):
        assert isinstance(next(iter(base_corpus)), collections.abc.Sequence)
        super(SlidingWindowAndGlobal, self).__init__(base_corpus=base_corpus.metadata,
                                                     left_ctx_size=left_ctx_size,
                                                     right_ctx_size=right_ctx_size,
                                                     verbose=verbose)
        self.base_corpus = base_corpus
        self.left_ctx_size = left_ctx_size
        self.right_ctx_size = right_ctx_size

    def _generate_samples(self):
        for sample_elems in self.base_corpus:
            for _, current, ctx in iter_sliding_window(sample_elems,
                                                       self.left_ctx_size,
                                                       self.right_ctx_size):
                yield dict(current=current,
                           context=ctx,
                           global_context=list(sample_elems))
