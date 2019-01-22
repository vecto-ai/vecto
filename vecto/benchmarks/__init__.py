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


def list_benhcmarks(benchmarks):
    print("available benchmarks:")
    for i in benchmarks:
        print(i)


def _run(args=None):
    # TODO: load them from modules themselves
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
    args, unknownargs = parser.parse_known_args(args)
    if args.name == "help":
        list_benhcmarks(available_benchmarks)
        return

    # TODO: implement running set of benchmarks defined in config
    # if args.name == "all":
        # print("running all benchmarks")

    if args.name in available_benchmarks:
        print("running ", args.name)
        mod = importlib.import_module("vecto.benchmarks." + args.name)
        run = getattr(mod, 'run')
        run(unknownargs)
    else:
        print("unknown benchmark name", args.name)
        list_benhcmarks(available_benchmarks)
        exit(-1)
