import argparse
from vecto.benchmarks.analogy import *
from vecto.config import load_config
from vecto.embeddings import load_from_dir


def main():
    # print("analogy cli")
    config = load_config()
    # print(config)
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument("--path_out", help="destination folder to save results")
    args = parser.parse_args()
    embeddings = load_from_dir(args.embeddings)
    benchmark = LRCos()
    results = benchmark.get_result(embeddings, args.dataset)


if __name__ == "__main__":
    main()
