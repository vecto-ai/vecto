import abc
import os
from vecto.utils.metadata import WithMetaData
from vecto.embeddings import load_from_dir
from vecto.utils.data import save_json, print_json
from vecto.utils import get_time_str


class Benchmark():
    # TODO: define proper interface

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_result(self, embeddings, path_dataset):
        raise NotImplementedError

    def run_with_args(self, args):
        embeddings = load_from_dir(args.embeddings)
        print("SHAPE:", embeddings.matrix.shape)
        print("vocab sieze:", embeddings.vocabulary.cnt_words)
        results = self.get_result(embeddings, args.dataset)
        if args.path_out:
            if os.path.isdir(args.path_out) or args.path_out.endswith("/"):
                dataset = os.path.basename(os.path.normpath(args.dataset))
                timestamp = get_time_str()
                task = results[0]["experiment_setup"]["task"]
                name_file_out = os.path.join(args.path_out,
                                             task,
                                             dataset,
                                             timestamp,
                                             "results.json")
                save_json(results, name_file_out)
            else:
                save_json(results, args.path_out)
        else:
            print_json(results)
