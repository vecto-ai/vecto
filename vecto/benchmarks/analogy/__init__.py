import argparse
import logging
import os
from .analogy import Analogy
from vecto.utils.data import save_json, print_json
# from vecto.config import load_config
from vecto.embeddings import load_from_dir
from vecto.utils import get_time_str

logging.basicConfig(level=logging.DEBUG)


def run(args):
    # config = load_config()
    # print(config)
    print(args)
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument("--method", help="analogy solving method", default="LRCos")
    parser.add_argument("--path_out", help="destination folder to save results")
    args = parser.parse_args(args)
    embeddings = load_from_dir(args.embeddings)
    # print("embeddings", embeddings)
    benchmark = Analogy(method=args.method)
    results = benchmark.get_result(embeddings, args.dataset)
    if args.path_out:
        if os.path.isdir(args.path_out) or args.path_out.endswith("/"):
            dataset = os.path.basename(os.path.normpath(args.dataset))
            timestamp = get_time_str()
            name_file_out = os.path.join(args.path_out,
                                         "analogical_reasoning",
                                         dataset,
                                         args.method,
                                         timestamp,
                                         "results.json")
            save_json(results, name_file_out)
        else:
            save_json(results, args.path_out)
    else:
        print_json(results)
