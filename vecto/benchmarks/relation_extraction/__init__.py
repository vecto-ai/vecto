import argparse
from .relation_extraction import Relation_extraction


def run(extra_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("dataset")

    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--nb_filter', '-nf', type=int, default=100,
                        help='filter number')
    parser.add_argument('--filter_length', '-fl', type=int, default=3,
                        help='filter length')
    parser.add_argument('--hidden_dims', '-hd', type=int, default=100,
                        help='D')
    parser.add_argument('--position_dims', '-pd', type=int, default=100,
                        help='D')
    parser.add_argument("--path_out",
                        default=False,
                        help="destination folder to save results")
    args = parser.parse_args(extra_args)
    relation_extraction = Relation_extraction(batchsize=args.batchsize,
                                              epoch=args.epoch,
                                              nb_filter=args.nb_filter,
                                              filter_length=args.filter_length,
                                              hidden_dims=args.hidden_dims,
                                              position_dims=args.position_dims)
    relation_extraction.run_with_args(args)
