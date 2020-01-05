import argparse
from .categorization import KMeansCategorization as Benchmark
from .categorization import purity_score
from vecto.embeddings import load_from_dir
from vecto.utils.data import save_json, print_json
from vecto.utils import get_time_str


def select_method(key):
    options = {}
    # if key == 'SpectralCategorization':
    #     method = SpectralCategorization(options)
    if key == 'KMeansCategorization':
        method = KMeansCategorization(options)
    else:
        raise RuntimeError('The method name was not recognized.')
    return method


def add_extra_args(parser):
    parser.add_argument('embeddings')
    parser.add_argument('dataset')
    # TODO: move method selection to benchmark class
#   parser.add_argument('--method', help='Categorization method', default='KMeansCategorization')
    # args = parser.parse_args(extra_args)
    # embeddings = load_from_dir(args.embeddings)
    # benchmark = select_method(args.method)
    # results = benchmark.get_result(embeddings, args.dataset)
    # if args.path_out:
    #     # TODO: this does not seem to work if the dir does not exist
    #     # let us always assume dir, clean this up later if no better idea 
    #     # if path.isdir(args.path_out) or args.path_out.endswith('/'):
    #     dataset = path.basename(path.normpath(args.dataset))
    #     timestamp = get_time_str()
    #     name_file_out = path.join(args.path_out,
    #                               dataset,
    #                               args.method,
    #                               timestamp,
    #                               'results.json')
    #     save_json(results, name_file_out)
    #     # else:
    #         # save_json(results, args.path_out)
    # else:
    #     print_json(results)
