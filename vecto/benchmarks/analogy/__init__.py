"""Benchmark on word analogy

.. autosummary::
    :toctree: _autosummary

    analogy
"""

import argparse
import logging
from .analogy import Analogy
# from vecto.config import load_config
from vecto.embeddings import load_from_dir

logging.basicConfig(level=logging.DEBUG)


def run(args):
    # config = load_config()
    # print(config)
    # print(args)
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument("--method", help="analogy solving method", default="LRCos")
    parser.add_argument("--path_out", help="destination folder to save results")
    args = parser.parse_args(args)
    benchmark = Analogy(method=args.method)
    benchmark.run_with_args(args)
