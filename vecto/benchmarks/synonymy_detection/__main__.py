import argparse
import json
import logging

from vecto.utils.data import save_json
from vecto.benchmarks.synonymy_detection import *
from vecto.embeddings import load_from_dir

logging.basicConfig(level=logging.DEBUG)


def print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=4, sort_keys=False))


def select_method(key):
    options = {}
    if key == 'CosineDistance':
        method = CosineDistance(options)
    else:
        raise RuntimeError('The method name was not recognized.')
    return method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings')
    parser.add_argument('dataset')
    parser.add_argument('--method', help='Synonymy detection method', default='CosineDistance')
    parser.add_argument('--path_out', help='Destination folder to save the results')
    args = parser.parse_args()
    embeddings = load_from_dir(args.embeddings)
    benchmark = select_method(args.method)
    results = benchmark.get_result(embeddings, args.dataset)
    if args.path_out:
        if path.isdir(args.path_out) or args.path_out.endswith('/'):
            dataset = path.basename(path.normpath(args.dataset))
            name_file_out = path.join(args.path_out, dataset, args.method, 'results.json')
            save_json(results, name_file_out)
        else:
            save_json(results, args.path_out)
    else:
        print_json(results)


if __name__ == '__main__':
    main()
