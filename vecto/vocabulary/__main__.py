import os
import argparse
from .vocabulary import create_ngram_tokens_from_dir, create_from_annotated_dir
from .vocabulary import create_from_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-t', choices=['normal', 'annotated', 'ngram_tokens'],
                        default='normal',
                        help='vocab type,')
    parser.add_argument('--min_ngram', '-minn', default=2, type=int,
                        help='minimal number of ngrams')
    parser.add_argument('--max_ngram', '-maxn', default=3, type=int,
                        help='minimal number of ngrams')
    parser.add_argument('--min_frequency', '-minf', default=100, type=int,
                        help='minimal number of ngrams')
    parser.add_argument('--context_representation', '-cp', choices=['word', 'deps', 'ne', ],
                        default='word',
                        help='context representation'
                             'the annotated corpus is required')
    parser.add_argument('--path_corpus', help='path to the corpus', required=True)
    parser.add_argument('--path_out', help='path to save vocab', required=True)

    args = parser.parse_args()
    return args


def run(args):
    print(args.type)
    if args.type == "normal":
        v = create_from_path(args.path_corpus, args.min_frequency)
        v.save_to_dir(os.path.join(args.path_out, args.type))
    if args.type == "annotated":
        v = create_from_annotated_dir(args.path_corpus, args.min_frequency, args.context_representation)
        v.save_to_dir(os.path.join(args.path_out, args.type, args.context_representation))
    if args.type == "ngram_tokens":
        v = create_ngram_tokens_from_dir(args.path_corpus, args.min_ngram, args.max_ngram, args.min_frequency)
        v.save_to_dir(os.path.join(args.path_out, args.type, str(args.min_ngram), str(args.max_ngram)))


def main():
    args = parse_args()
    # print(args)
    run(args)


if __name__ == "__main__":
    main()
