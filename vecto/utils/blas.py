import numpy as np
import functools
import scipy.sparse.linalg


def identity(x):
    return x


def normed(v, ord=None):
    return v / np.linalg.norm(v, ord=ord)


def make_normalizer(ord='unnormed'):
    if ord == 'unnormed':
        return identity
    else:
        return functools.partial(normed, ord=ord)


def normalize_sparse(m):
    norm = scipy.sparse.linalg.norm(m, axis=1)[:, None]
    m.data /= norm.repeat(np.diff(m.indptr))
