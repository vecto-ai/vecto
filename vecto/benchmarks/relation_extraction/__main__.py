import argparse
import json
import logging
import os

from vecto.utils.data import save_json
from vecto.benchmarks.relation_extraction import Relation_extraction
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

    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--nb_filter', '-nf', type=int, default=100,
                        help='filter number')
    parser.add_argument('--filter_length', '-fl', type=int, default=3,
                        help='filter length')
    parser.add_argument('--hidden_dims', '-hd', type=int, default=100,
                        help='D')
    parser.add_argument('--position_dims', '-pd', type=int, default=100,
                        help='D')
    parser.add_argument("--path_out", default=False, help="destination folder to save results")
    args = parser.parse_args()
    embeddings = load_from_dir(args.embeddings)
    # print("embeddings", embeddings)
    # print(args.normalize)
    relation_extraction = Relation_extraction(batchsize=args.batchsize,
                                              epoch=args.epoch,
                                              nb_filter=args.nb_filter,
                                              filter_length=args.filter_length,
                                              hidden_dims=args.hidden_dims,
                                              position_dims=args.position_dims,)
    results = relation_extraction.get_result(embeddings, args.dataset)
    if args.path_out:
        if os.path.isdir(args.path_out) or args.path_out.endswith("/"):
            dataset = os.path.basename(os.path.normpath(args.dataset))
            name_file_out = os.path.join(args.path_out, dataset, "results.json")
            save_json(results, name_file_out)
        else:
            save_json(results, args.path_out)
    else:
        print_json(results)


if __name__ == "__main__":
    main()
