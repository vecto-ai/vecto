"""Benchmark on word analogy

.. autosummary::
    :toctree: _autosummary

    analogy
"""

# import logging
from .analogy import Analogy as Benchmark


# logging.basicConfig(level=logging.DEBUG)


def add_extra_args(parser):
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument("--method",
                        help="analogy solving method",
                        default="LRCos")
