"""Text classification benchmark.

    One of the pre-defined models is trained to convergence
    to predict labels for text fragments in a provided dataset.
    Sentiment analysis is an example of text classification task.

.. autosummary::
    :toctree: _autosummary

    text_classification
"""

import argparse
import os
from vecto.utils.data import save_json, print_json
from .text_classification import Text_classification
from vecto.embeddings import load_from_dir


def run(extra_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of layers of RNN or MLP following CNN')
    parser.add_argument('--dropout', '-d', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--model', '-model', default='cnn',
                        choices=['cnn', 'rnn', 'bow'],
                        help='Name of encoder model type')
    parser.add_argument("--path_out", default=False, help="destination folder to save results")
    args = parser.parse_args(extra_args)
    embeddings = load_from_dir(args.embeddings)
    text_classification = Text_classification(batchsize=args.batchsize, epoch=args.epoch, gpu=args.gpu,
                                              layer=args.layer, dropout=args.dropout, model=args.model)
    text_classification.run_with_args(args)
