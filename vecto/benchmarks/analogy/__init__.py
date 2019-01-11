import argparse
import json
import logging
import os
from .analogy import ThreeCosAvg, ThreeCosMul, ThreeCosMul2, LinearOffset, LRCos, PairDistance
from vecto.utils.data import save_json, print_json
from vecto.benchmarks.analogy import ThreeCosAvg, ThreeCosMul, LinearOffset, LRCos
# from vecto.config import load_config
from vecto.embeddings import load_from_dir


logging.basicConfig(level=logging.DEBUG)


def select_method(key):
    options = {}
    if key == "3CosAvg":
        method = ThreeCosAvg(options)
    #elif key == "SimilarToAny":
    #    method = SimilarToAny(options)
    #elif key == "SimilarToB":
    #    method = SimilarToB(options)
    elif key == "3CosMul":
        method = ThreeCosMul(options)
    elif key == "3CosAdd":
        method = LinearOffset(options)
    #elif key == "PairDistance":
    #    method = PairDistance(options)
    elif key == "LRCos" or key == "SVMCos":
        method = LRCos(options)
    else:
        raise RuntimeError("method name not recognized")
    return method


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
    benchmark = select_method(args.method)
    results = benchmark.get_result(embeddings, args.dataset)
    if args.path_out:
        if os.path.isdir(args.path_out) or args.path_out.endswith("/"):
            dataset = os.path.basename(os.path.normpath(args.dataset))
            timestamp = results[0]["experiment_setup"]["timestamp"]
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
