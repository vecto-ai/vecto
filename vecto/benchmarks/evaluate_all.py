"""

this script evaluate all available benchmarks in vecto

"""
import argparse
import os
import sys
import json
from multiprocessing import Pool
from multiprocessing import Process
import copy
from vecto.benchmarks.similarity import Similarity
from vecto.benchmarks.sequence_labeling import Sequence_labeling

from vecto.benchmarks.analogy import visualize as analogy_visualize
from vecto.benchmarks.similarity import visualize as similarity_visualize
from vecto.benchmarks.analogy import *
from vecto.benchmarks.text_classification import Text_classification
from vecto.benchmarks import text_classification
from vecto.embeddings import load_from_dir
from vecto.benchmarks.fetch_benchmarks import fetch_benchmarks
from os import path

test_word_similarity = True
test_word_analogy = False
test_sequence_labeling = False
test_text_classification = False
test_machine_translation = False
test_language_modeling = False
default_path_root_dataset = "/home/users/bofang/work/data/NLP/datasets/"
default_folder_name_keys = ['task', 'dataset', 'method', 'measurement']


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_root_dataset', default=default_path_root_dataset)
    parser.add_argument('--processes', type=int, help='processes to run the evaluation', default=1)
    parser.add_argument('--path_vector', help='path to the vector', required=True)
    parser.add_argument('--path_output', help='path to the output', required=True)

    args = parser.parse_args(args)
    return args


def get_file_name(json_data):
    # print(json.dumps(json_data, sort_keys=True, indent=4))
    name = ''
    for key in default_folder_name_keys:
        if key in json_data:
            name += '_' + str(json_data[key])
    name = name[1:]
    return name


def write_results(args, results):
    for result in results:
        if 'result' in result:
            print(result['result'], end=', ')
    print('')
    for result in results:
        file_name = get_file_name(result['experiment_setup'])
        path = args.path_vectors
        path = os.path.join(path, file_name + ".json")
        with open(path, 'w') as output:
            output.write(json.dumps(result, sort_keys=True, indent=4))


def worth_evaluate(path):
    if 'context' in path:
        return False

    has_npy = False
    for f in os.listdir(path):
        if f.endswith(".npy"):
            has_npy = True

    if not os.path.isfile(os.path.join(path, "vectors.h5p")) \
            and not os.path.isfile(os.path.join(path, "vectors.txt")) \
            and not os.path.isfile(os.path.join(path, "vectors.bin")) \
            and not os.path.isfile(os.path.join(path, "vectors.vec")) \
            and not has_npy:
        return False
    return True


def run(args):
    args.path_vectors = args.path_vector

    if not worth_evaluate(args.path_vector):
        return

    embs = load_from_dir(args.path_vector)

    print(args.path_vector)
    if test_word_similarity is True:
        similarity = Similarity()
        results = similarity.get_result(embs, os.path.join(args.path_root_dataset, "similarity"))
        write_results(args, results)

    if test_word_analogy is True:
        analogy = LRCos()

        results = analogy.get_result(embs, os.path.join(args.path_root_dataset, "analogies", "Google_dir"),
                                     group_subcategory=True)
        write_results(args, results)

        # result = analogy.get_result(embs, os.path.join(args.path_root_dataset, "analogies", "BATS_3.0"), group_subcategory=True)
        # write_results(args, result)

    if test_sequence_labeling is True:
        sequence_labeling = Sequence_labeling()

        for subtask in ['chunk', 'pos', 'ner']:  # , 'chunk', 'pos', 'ner'
            results = sequence_labeling.get_result(embs,
                                                   os.path.join(args.path_root_dataset, "sequence_labeling", subtask))
            write_results(args, results)


def main(args=None):
    args = parse_args(args)
    path_vector = args.path_vector

    run(args)

    pool = Pool(processes=args.processes)

    argsList = []

    for root, dirs, files in os.walk(path_vector, topdown=False):
        for name in dirs:
            # print(name)
            args.path_vector = os.path.join(root, name)
            a = copy.deepcopy(args)

            if not worth_evaluate(a.path_vector):
                continue

            argsList.append(a)
            # run(args)
    print(len(argsList))
    pool.map(run, argsList)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
