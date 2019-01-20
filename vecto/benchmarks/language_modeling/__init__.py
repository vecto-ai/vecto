import argparse
import logging
import os

from vecto.utils.data import save_json, print_json
from vecto.embeddings import load_from_dir
from .language_modeling import Language_modeling

logging.basicConfig(level=logging.DEBUG)


def run(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--test", default=True,
                        help='use small test dataset')
    parser.add_argument("--method", default='lstm', choices=['lr', '2FFNN', 'lstm'],
                        help='name of method')
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument("--path_out", default=False, help="destination folder to save results")
    args = parser.parse_args(args)
    embeddings = load_from_dir(args.embeddings)
    # print("embeddings", embeddings)
    language_modeling = Language_modeling(normalize=args.normalize, window_size=args.window_size, method=args.method, test=args.test)
    results = language_modeling.get_result(embeddings)
    if args.path_out:
        if os.path.isdir(args.path_out) or args.path_out.endswith("/"):
            name_file_out = os.path.join(args.path_out, "language_modeling", "results.json")
            save_json(results, name_file_out)
        else:
            save_json(results, args.path_out)
    else:
        print_json(results)
