"""Collection of benchmarks and downstream tasks on embeddings

.. autosummary::
    :toctree: _autosummary

    analogy

"""

import argparse
import importlib


def list_benhcmarks():
    print("available benchmarks:")
    # TODO: list benchmarks


def _run(args=None):
    parser = argparse.ArgumentParser(
        description='run benchmarks',
        add_help=True,
        usage="vecto benchmark [name]")

    parser.add_argument('name', help='Subcommand to run')
    args, unknownargs = parser.parse_known_args(args)
    if args.name == "help":
        list_benhcmarks()
        return
    # if args.name == "all":
        # print("running all benchmarks")

    options = {}

    if args.name == "analogy":
        print("running analogy")
        from .analogy import run
        run(unknownargs)
    elif args.name == "categorization":
        print("running categorization")
        from .categorization import run
        run(options, unknownargs)
    elif args.name == "similarity":
        print("running similarity")
        from .similarity import run
        run(options, unknownargs)
    elif args.name == "sequence_labelling":
        print("running sequence labelling")
        from .sequence_labeling import run
        run(options, unknownargs)
    elif args.name == "text_classification":
        print("running sequence labelling")
        from .text_classification import run
        run(options, unknownargs)
    else:
        print("unknown benchmark name", args.name)
        list_benhcmarks()
        exit(-1)
    # check if all is specified - then run all
    # if benchmark name matches - run corresponding module
    # list all available benchmarks
