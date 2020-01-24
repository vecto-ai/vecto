"""Benchmark on word analogy

.. autosummary::
    :toctree: _autosummary

    analogy
"""

# import logging
from .analogy import Analogy as Benchmark
import numpy as np

# logging.basicConfig(level=logging.DEBUG)


def add_extra_args(parser):
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument("--method",
                        help="analogy solving method",
                        default="LRCos")


# TODO: move this to proper location, reuse between senchmarks
def get_mean_reciprocal_rank(results):
    mean_reciprocal_rank=np.mean([(lambda r : 0 if r<=0 else 1/r) (experiment["rank"]) for category in results for experiment in category["details"] ])
    return mean_reciprocal_rank


def get_mean_accuracy(results):
    mean_accuracy=np.mean([experiment["rank"]==0 for category in results for experiment in category["details"] ])
    return mean_accuracy
