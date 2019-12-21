import fnmatch
import os
import tarfile
from vecto.utils.metadata import WithMetaData
from .io import fetch_file

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

    def file_iterator(self):
        for root, _, filenames in os.walk(self.path):
            for filename in fnmatch.filter(sorted(filenames), '*'):
                if filename.endswith('json'):
                    continue
                yield(os.path.join(root, filename))


def download_index():
    # TODO: get paths from config module
    dir_temp = "/tmp/vecto/tmp"
    os.makedirs(dir_temp, exist_ok=True)
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
            tar.extract(member, dir_temp)
        


def get_dataset(name):
    # TODO: get dataset dir from config
    dir_datasets = "/home/blackbird/.vecto/datasets"
    path_dataset = os.path.join(dir_datasets, name)
    dataset = Dataset(path_dataset)
    return dataset
    # TODO: check if it seats locally
    # TODO: download
