import argparse
import json
import logging

from vecto.utils.data import save_json
from vecto.benchmarks.analogy import *
from vecto.config import load_config
from vecto.embeddings import load_from_dir

logging.basicConfig(level=logging.DEBUG)


def print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=4, sort_keys=False))


def main():
    # config = load_config()
    # print(config)
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument("--path_out", help="destination folder to save results")
    args = parser.parse_args()
    embeddings = load_from_dir(args.embeddings)
    benchmark = LRCos()
    results = benchmark.get_result(embeddings, args.dataset)
    if args.path_out:
        if os.path.isdir(args.path_out):
            filename = os.path.join(args.path_out, "results.json")
            save_json(results, filename)
        else:
            save_json(results, args.path_out)
    else:
        print_json(results)


if __name__ == "__main__":
    main()
