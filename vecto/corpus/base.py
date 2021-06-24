import abc

from vecto.utils.metadata import WithMetaData
from vecto.utils.tqdm_utils import get_tqdm


class BaseIterator(WithMetaData):
    """
    Base class for all corpora and iterators.
    Responsible for base logic like metadata collection, __len__,
    iteration, tqdm progressbar etc.
    """

    def __init__(self, verbose=False, **metadata_kwargs):
        super(BaseIterator, self).__init__(**metadata_kwargs)
        self._verbose = verbose

    def __iter__(self):
        for elem in self._generate_samples_outer():
            yield elem

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
        return gen
