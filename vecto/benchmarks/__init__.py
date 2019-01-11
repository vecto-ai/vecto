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
        usage="vecto benchmark [all|<name>]")

    parser.add_argument('name', help='Subcommand to run')
    args, unknownargs = parser.parse_known_args(args)
    if args.name == "help":
        list_benhcmarks()

    if args.name == "all":
        print("running all benchmarks")

    
    # check if all is specified - then run all
    # if benchmark name matches - run corresponding module
    # list all available benchmarks
