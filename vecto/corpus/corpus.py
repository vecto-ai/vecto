import fnmatch
import os

import logging
from .base import BaseCorpus
from .iterators import FileTokenIterator
from .tokenization import DEFAULT_TOKENIZER


logger = logging.getLogger(__name__)


class FileCorpus(BaseCorpus):
    def __init__(self, filename, verbose=1):
        super(FileCorpus, self).__init__(base_path=filename,
                                         verbose=verbose)
        self.filename = filename

    def _generate_samples(self):
        yield self.filename


def FileTokenCorpus(path, *args, **kwargs):
    return FileTokenIterator(FileCorpus(path), *args, **kwargs)


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


class DirCorpus(BaseCorpus):
    def __init__(self, dirname, verbose=1):
        super(DirCorpus, self).__init__(base_path=dirname,
                                        verbose=verbose)
        self.dirname = dirname

    def _generate_samples(self):
        for root, _, files in os.walk(self.dirname, followlinks=True):
            for good_fname in fnmatch.filter(files, "*"):
                logger.info("processing " + os.path.join(root, good_fname))
                yield good_fname


def DirTokenCorpus(path, *args, **kwargs):
    return FileTokenIterator(DirCorpus(path), *args, **kwargs)


class LimitedCorpus(BaseCorpus):
    def __init__(self, base, limit=1000, verbose=1):
        super(LimitedCorpus, self).__init__(base=base.meta,
                                            verbose=verbose)
        self.samples = []
        for i, s in enumerate(base):
            if i >= limit:
                break
            self.samples.append(s)
        self.metadata['samples_count'] = len(self.samples)

    def _generate_samples(self):
        for s in self.samples:
            yield s
