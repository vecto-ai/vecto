from .relation_extraction import Relation_extraction as Benchmark


def add_extra_args(parser):
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
