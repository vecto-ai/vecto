import fnmatch
import os
import abc

from vecto.utils.metadata import WithMetaData
from vecto.utils.tqdm_utils import get_tqdm
import logging

logger = logging.getLogger(__name__)


class BaseCorpus(WithMetaData):
    def __init__(self, verbose=1, **metadata_kwargs):
        super(BaseCorpus, self).__init__(**metadata_kwargs)
        self._verbose = verbose

    def __iter__(self):
        for s in self._generate_samples_outer():
            yield s

    def __len__(self):
        return self.metadata.get('samples_count', 0)

    def _generate_samples_outer(self):
        gen = self._generate_samples()
        if self._verbose > 0:
            cur_len = len(self)
            if cur_len is None:
                return get_tqdm(gen)
            else:
                return get_tqdm(gen, total=cur_len)
        else:
            return gen

    @abc.abstractmethod
    def _generate_samples(self):
        pass


class FileCorpus(BaseCorpus):
    def __init__(self, filename, verbose=1):
        super(FileCorpus, self).__init__(base_path=filename,
                                         verbose=verbose)
        self.filename = filename

    def _generate_samples(self):
        yield self.filename


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
