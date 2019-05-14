"""Loading and training for embeddings

.. autosummary::
    :toctree: _autosummary

    base
    dense

"""

import os
import logging
import numpy as np
import vecto.embeddings.dense
from vecto.embeddings.dense import WordEmbeddingsDense
from .legacy_w2v import ModelW2V
from vecto.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def load_from_dir(path):
    """Automatically detects embeddings format and loads

    Args:
        path: directory where embeddings are stores

    Returns:
        Instance of appropriate Model-based class
    """
#    if os.path.isfile(os.path.join(path, "cooccurrence_csr.h5p")):
#        logger.info("detected as sparse explicit in hdf5")
#        result = ModelSparse()
#        result.load_from_hdf5(path)
#        result.load_metadata(path)
#        return result
#    if os.path.isfile(os.path.join(path, "bigrams.data.bin")):
#        logger.info("detected as sparse in vecto legacy format")
#        result = ModelSparse()
#        result.load(path)
#        result.load_metadata(path)
#        return result

#    if os.path.isfile(os.path.join(path, "sgns.words.npy")):
#        result = ModelLevy()
#        logger.info("this is Levi")
#        result.load_from_dir(path)
#        result.load_metadata(path)
#        return result
#     if os.path.isfile(os.path.join(path, "vectors.npy")):
#         result = ModelNumbered()
#         logger.info("detected as dense ")
#         result.load_npy(path)
#         result.load_metadata(path)
#         return result
    if os.path.isfile(os.path.join(path, "vectors.h5p")):
        result = vecto.embeddings.dense.WordEmbeddingsDense()
        logger.info("detected as vecto format ")
        result.load_hdf5(path)
        result.load_metadata(path)
        # TODO: remove this hack after we re-train w2v without OOV rows
        extra = result.matrix.shape[0] - result.vocabulary.cnt_words
        result.matrix = result.matrix[extra:]
        return result

    result = vecto.embeddings.dense.WordEmbeddingsDense()
    files = os.listdir(path)
    for f in files:
        if f.endswith(".gz") or f.endswith(".bz") or f.endswith(".txt") or f.endswith(".vec"):
            logger.info(path + "Detected plain text format")
            result.load_from_text(os.path.join(path, f))
            result.load_metadata(path)
            return result
        if f.endswith(".npy"):
            logger.info("Detected numpy format")
            result.matrix = np.load(os.path.join(path, f))
            result.vocabulary = Vocabulary()
            result.vocabulary.load(path)
            result.load_metadata(path)
            # TODO: remove this hack after we re-train w2v without OOV rows
            result.matrix = result.matrix[:result.vocabulary.cnt_words]
            return result
        if any(file.endswith('bin') for file in os.listdir(path)):
            result = ModelW2V()
            logger.info("Detected w2v original binary format")
            result.load_from_dir(path)
            result.load_metadata(path)
            return result
#        if f.startswith("words") and f.endswith(".npy") \
#               and os.path.isfile(os.path.join(path, f.replace(".npy", ".vocab"))):
#            result = Model_Fun()
#            result = ModelLevy()
#            logger.info("Detected VSM in npy and vocab in plain text file format")
#            result.load_from_dir(path, f[: -4])
#            result.load_metadata(path)
#            return result

    raise RuntimeError("Cannot detect the format of this VSM")

