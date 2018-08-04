import argparse
import json
import logging
import os

from vecto.utils.data import save_json
from vecto.benchmarks.language_modeling import Language_modeling
from vecto.embeddings import load_from_dir

logging.basicConfig(level=logging.DEBUG)


def print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=4, sort_keys=False))


def main():
    # config = load_config()
    # print(config)
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--test", default=True,
                        help='use small test dataset')
    parser.add_argument("--method", default='lstm', choices=['lr', '2FFNN', 'lstm'],
                        help='name of method')
    parser.add_argument("--path_out", default=False, help="destination folder to save results")
    args = parser.parse_args()
    embeddings = load_from_dir(args.embeddings)
    # print("embeddings", embeddings)
    language_modeling = Language_modeling(window_size=args.window_size, method=args.method, test=args.test)
    results = language_modeling.get_result(embeddings)
    if args.path_out:
        if os.path.isdir(args.path_out) or args.path_out.endswith("/"):
            name_file_out = os.path.join(args.path_out, "language_modeling", "results.json")
            save_json(results, name_file_out)
        else:
            save_json(results, args.path_out)
    else:
        print_json(results)


if __name__ == "__main__":
    main()
