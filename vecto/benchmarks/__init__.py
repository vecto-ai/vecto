"""Collection of benchmarks and downstream tasks on embeddings

.. autosummary::
    :toctree: _autosummary

    analogy
    categorization
    language_modeling
    outliers
    relation_extraction
    sequence_labeling
    similarity
    synonymy_detection
    text_classification

"""

import argparse
import importlib
from vecto.embeddings import load_from_dir
from vecto.data import Dataset
import os
from vecto.utils.data import save_json, print_json
from vecto.utils import get_time_str


def list_benhcmarks(benchmarks):
    print("available benchmarks:")
    for i in benchmarks:
        print(i)


def choose_benchmark(args):
    # TODO: load benchmark names from modules themselves
    available_benchmarks = []
    available_benchmarks.append("analogy")
    available_benchmarks.append("categorization")
    available_benchmarks.append("language_modeling")
    available_benchmarks.append("relation_extraction")
    available_benchmarks.append("similarity")
    available_benchmarks.append("sequence_labeling")
    available_benchmarks.append("text_classification")

    parser = argparse.ArgumentParser(
        description='run benchmarks',
        add_help=True,
        usage="vecto benchmark [name]")

    parser.add_argument('name', help='Subcommand to run')
    args, remaining_args = parser.parse_known_args(args)
    if args.name == "help":
        list_benhcmarks(available_benchmarks)
        return
    # TODO: implement running set of benchmarks defined in config
    # if args.name == "all":
        # print("running all benchmarks")

    if args.name in available_benchmarks:
        #print('ramaining args')
        #print(remaining_args)
        run_benchmark_by_name(args.name, remaining_args)
    else:
        print("unknown benchmark name", args.name)
        list_benhcmarks(available_benchmarks)
        exit(-1)


def save_results(results, path_out, dataset_name):
    # create subdirs unless explicitly asked to not do so
    # TODO: add submodules to append to path
    timestamp = get_time_str()
    if isinstance(results, list):
        task = results[0]["experiment_setup"]["task"]
    else:
        task = results["experiment_setup"]["task"]
    task = task.replace(" ", "_")
    name_file_out = os.path.join(path_out,
                                 task,
                                 dataset_name,
                                 timestamp,
                                 "results.json")
    save_json(results, name_file_out)


def run_benchmark_by_name(name, args):
    print(name, args)
    print("running ", name)
    mod = importlib.import_module("vecto.benchmarks." + name)
    parser = argparse.ArgumentParser()
    add_extra_args = getattr(mod, 'add_extra_args')
    add_extra_args(parser)
    parser.add_argument("--path_out",
                        default=None,
                        help="destination folder to save results")
    args = parser.parse_args(args)
    dict_args = vars(args)
    embeddings = load_from_dir(args.embeddings)
    # TODO: this is ugly hack, do subparsers or something
    if name == "language_modeling":
        dataset = Dataset("/tmp/")
        dataset.name = "ptb"
    else:
        dataset = Dataset(args.dataset)
        dict_args.pop("dataset")

    dict_args.pop("embeddings")
    # TODO: not sure if all banchmarks use dataset arg
    path_out = dict_args.pop("path_out")
    Benchmark = getattr(mod, "Benchmark")
    benchmark = Benchmark(**dict_args)

    print("SHAPE:", embeddings.matrix.shape)
    print("vocab size:", embeddings.vocabulary.cnt_words)
    results = benchmark.run(embeddings, dataset)
    if path_out:
        save_results(results, path_out, dataset.metadata["name"])
    else:
        print_json(results)


def run_benchmarks_cli(args=[]):
    choose_benchmark(args)
