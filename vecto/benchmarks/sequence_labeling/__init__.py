import argparse
import os
from vecto.embeddings import load_from_dir
from vecto.utils.data import save_json, print_json
from .sequence_labeling import Sequence_labeling


def run(options, extra_args):

    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings")
    parser.add_argument("dataset")
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--method", default='lr', choices=['lr', '2FFNN'],
                        help='name of method')
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument("--path_out", default=False, help="destination folder to save results")
    args = parser.parse_args(extra_args)
    embeddings = load_from_dir(args.embeddings)
    # print("embeddings:", embeddings)
    sequence_labeling = Sequence_labeling(normalize=args.normalize, method=args.method, window_size=args.window_size)
    results = sequence_labeling.get_result(embeddings, args.dataset)
    if args.path_out:
        if os.path.isdir(args.path_out) or args.path_out.endswith("/"):
            dataset = os.path.basename(os.path.normpath(args.dataset))
            name_file_out = os.path.join(args.path_out, dataset, "results.json")
            save_json(results, name_file_out)
        else:
            save_json(results, args.path_out)
    else:
        print_json(results)
