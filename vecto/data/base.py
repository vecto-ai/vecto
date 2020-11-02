
import fnmatch
import os
from pathlib import Path
import tarfile
from zipfile import ZipFile
import logging
import tempfile
import shutil
# from vecto.config import load_config
from vecto.utils.metadata import WithMetaData
from vecto.utils.data import load_json
from .io import fetch_file, read_first_col_is_label_format, read_tsv_label_last

logger = logging.getLogger(__name__)
# TODO: make config module-global
# config = load_config()
# TODO: get dataset dir from config
# TODO: use pathlib everywhere
dir_datasets = os.path.expanduser("~/.vecto/datasets")
dir_temp = os.path.join(tempfile.gettempdir(), "vecto", "tmp")
os.makedirs(dir_datasets, exist_ok=True)
os.makedirs(dir_temp, exist_ok=True)
resources = {}


class Dataset(WithMetaData):
    """
    Container class for stock datasets.
    Arguments:
        path (str): local path to place files
    """

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("test dataset dir does not exist:" + path)
        super().__init__(path)
        self.path = path
        if "name" not in self.metadata:
            # TODO: use proper path magic
            self.metadata["name"] = path.split("/")[-1]

    def file_iterator(self):
        for root, _, filenames in os.walk(self.path):
            for filename in fnmatch.filter(sorted(filenames), '*'):
                if filename.endswith('json'):
                    continue
                yield(os.path.join(root, filename))

    def _load_tsv(self, names):
        # TODO: decide what to do with char_basrd
        char_based = False
        for candidate_name in names:
            path_full_candidate = os.path.join(self.path, candidate_name)
            print(path_full_candidate)
            if os.path.isfile(path_full_candidate):
                # train = read_first_col_is_label_format(path_full_candidate, char_based=char_based)
                train = read_first_col_is_label_format(path_full_candidate)
                return train
        # test = read_first_col_is_label_format(os.path.join(self.path, 'test'),
        #                      char_based=char_based)
        raise RuntimeError("can not find dataset")

    def get_train(self):
        return self._load_tsv(["train", "train.tsv"])

    def get_test(self):
        return self._load_tsv(["dev", "dev.tsv", "test", "test.tsv"])


def download_index():
    logger.info("downloading index of resources")
    path_tar = os.path.join(dir_temp, "resources.tar")
    url_resources = "https://github.com/vecto-ai/vecto-resources/tarball/master/"
    fetch_file(url_resources, path_tar)
    with tarfile.open(path_tar) as tar:
        for member in tar.getmembers():
            parts = member.name.split("/")
            if len(parts) <= 1: 
                continue
            if parts[1] != "resources":
                continue
            member.path = os.path.join(*parts[1:])
            tar.extract(member, dir_datasets)


def gen_metadata_snippets(path):
    for sub in path.iterdir():
        if sub.name == "metadata.json":
            yield sub
        else:
            if sub.is_dir():
                yield from gen_metadata_snippets(sub)


def load_dataset_infos():
    for f_meta in gen_metadata_snippets(Path(dir_datasets)):
        # print("visiting", f_meta.parent)
        metadata = load_json(f_meta)
        if "name" in metadata:
            metadata["local_path"] = f_meta.parent
            resources[metadata["name"]] = metadata


def download_dataset_by_name(name, path_dataset):
    filename = resources[name]["url"].split("/")[-1]
    logger.debug("downloading ", filename)
    path_download_archive = Path(dir_temp) / filename
    if "url" not in resources[name]:
        raise RuntimeError(f"no URL to download dataset {name}")
    fetch_file(resources[name]["url"], path_download_archive)
    path_extracted = Path(dir_temp) / name
    with ZipFile(path_download_archive) as z:
        z.extractall(path_extracted)
    # TODO: make sure this returns topmost entry from the tree
    first_metadata_path = next(gen_metadata_snippets(path_extracted)).parent
    # print(first_metadata_path)
    for f in first_metadata_path.iterdir():
        if not (path_dataset / f.name).exists():
            shutil.move(str(f), str(path_dataset))


def is_dataset_downloaded(path_dataset):
    for f in path_dataset.iterdir():
        if f.name.endswith("metadata.json"):
            continue
        return True
    return False


def get_dataset_by_name(name):
    load_dataset_infos()
    if not resources:
        logger.info("index not found, forcing download")
        download_index()
        load_dataset_infos()
    # print(resources)
    if name in resources:
        path_dataset = resources[name]["local_path"]
    else:
        raise RuntimeError("Dataset %s not known" % name)
    if not is_dataset_downloaded(path_dataset):
        logger.info("only metadata is present, need to download")
        download_dataset_by_name(name, path_dataset)
    dataset = Dataset(path_dataset)
    return dataset
