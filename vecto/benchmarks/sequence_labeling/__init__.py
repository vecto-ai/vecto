import argparse
from .sequence_labeling import Sequence_labeling


def run(extra_args):

    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--method", default='lr', choices=['lr', '2FFNN'],
                        help='name of method')
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument("--path_out", default=False, help="destination folder to save results")
    args = parser.parse_args(extra_args)
    sequence_labeling = Sequence_labeling(normalize=args.normalize, method=args.method, window_size=args.window_size)
    sequence_labeling.run_with_args(args)
