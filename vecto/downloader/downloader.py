from zipfile import ZipFile
from requests import get
from os import path, walk, sep
from git import Repo
from git.exc import GitCommandError
from vecto.utils import metadata
from functools import reduce


class Downloader(object):
    def __init__(self, storage_dir=path.join('data', 'resources')):
        self.path_to_repo = 'https://github.com/vecto-ai/benchmarks.git'
        self.storage_dir = storage_dir
        self.dir_struct = {}

    def fetch_metadata(self):
        try:
            Repo.clone_from(self.path_to_repo, self.self.storage_dir)
        except GitCommandError:
            raise ValueError('Directory exists')

    def unarchive(self, input_dir, archive_type='.zip'):
        if archive_type == '.zip':
            with ZipFile(input_dir, '') as z:
                z.extractall(self.storage_dir)

    def download_resource(self, resource_name):
        resource_metadata = metadata.WithMetaData()
        resource_metadata.load_metadata(path.join(self.storage_dir, resource_name))
        self.fetch_file(resource_metadata['url'])

    def fetch_file(self, url, output_file='tmp'):
        with open(path.join(self.storage_dir, output_file), 'wb') as file:
            response = get(url)
            file.write(response.content)

    def get_directory_structure(self):
        rootdir = self.storage_dir.rstrip(sep)
        start = rootdir.rfind(sep) + 1
        for path, dirs, files in walk(rootdir):
            folders = path[start:].split(sep)
            subdir = dict.fromkeys(files)
            parent = reduce(dict.get, folders[:-1], self.dir_struct)
            parent[folders[-1]] = subdir
        return self.dir_struct
   