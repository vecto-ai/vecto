"""The model module that implements vocabulary.

.. autosummary::
    :toctree: _autosummary

"""

from .vocabulary import Vocabulary
from .vocabulary import create_from_path, create_ngram_tokens_from_dir, create_from_annotated_dir


def load(path):
    v = Vocabulary()
    v.load(path)
    return v
