import argparse
from .categorization import *
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


def run(extra_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings')
    parser.add_argument('dataset')
    parser.add_argument('--method', help='Categorization method', default='KMeansCategorization')
    parser.add_argument('--path_out', help='Destination folder to save the results')
    args = parser.parse_args(extra_args)
    embeddings = load_from_dir(args.embeddings)
    benchmark = select_method(args.method)
    results = benchmark.get_result(embeddings, args.dataset)
    if args.path_out:
        if path.isdir(args.path_out) or args.path_out.endswith('/'):
            dataset = path.basename(path.normpath(args.dataset))
            timestamp = get_time_str()
            name_file_out = path.join(args.path_out,
                                      dataset,
                                      args.method,
                                      timestamp,
                                      'results.json')
            save_json(results, name_file_out)
        else:
            save_json(results, args.path_out)
    else:
        print_json(results)
