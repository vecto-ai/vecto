import numpy as np
# import scipy.sparse.linalg


def normed(v):
    return v / np.linalg.norm(v)


# def normalize_sparse(m):
#     norm = scipy.sparse.linalg.norm(m, axis=1)[:, None]
#     m.data /= norm.repeat(np.diff(m.indptr))
