import argparse
from .similarity import Similarity


def run(extra_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--ignore_oov', dest='ignore_oov', action='store_true')
    parser.add_argument("--path_out", default=False, help="destination folder to save results")
    args = parser.parse_args(extra_args)
    similarity = Similarity(normalize=args.normalize, ignore_oov=args.ignore_oov)
    similarity.run_with_args(args)
