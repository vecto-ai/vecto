from .sequence_labeling import Sequence_labeling as Benchmark


def add_extra_args(parser):
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--method", default='lr', choices=['lr', '2FFNN'],
                        help='name of method')
    parser.add_argument('--normalize', dest='normalize', action='store_true')
