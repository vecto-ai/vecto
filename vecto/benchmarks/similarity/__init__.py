from .similarity import Similarity as Benchmark


def add_extra_args(parser):
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--ignore_oov', dest='ignore_oov', action='store_true')
