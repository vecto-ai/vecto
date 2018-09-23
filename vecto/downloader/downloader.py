from zipfile import ZipFile
from requests import get
from os import path, walk, sep, mkdir
from git import Repo, Git
from git.exc import GitCommandError
from vecto.utils.metadata import WithMetaData
from functools import reduce
from vecto.downloader.resources import Resources
from shutil import rmtree
from json import load

class Downloader:
    def __init__(self, storage_dir=path.join('data', 'resources')):
        self.path_to_repo = 'https://github.com/vecto-ai/vecto-resources.git'
        self.storage_dir = storage_dir
        self.resources = None
        self.git_repo = Git(self.storage_dir)
        self.full_resource_path = path.join('vecto-resources', 'resources')

    def fetch_metadata(self, replace=False):
        while True:
            try:
                self.git_repo.clone(self.path_to_repo)
                break
            except GitCommandError:
                if replace:
                    rmtree(self.storage_dir)
                    mkdir(self.storage_dir)
                else:
                    break
            except FileNotFoundError:
                mkdir(self.storage_dir)

    def unarchive(self, input_dir, archive_type='.zip'):
        if archive_type == '.zip':
            with ZipFile(input_dir, '') as z:
                z.extractall(self.storage_dir)

    def download_resource(self, resource_name):
        resource_metadata = WithMetaData()
        resource_metadata.load_metadata(path.join(self.storage_dir, 'vecto-resources', '/'.join(resource_name), 'metadata.json'))
        with open(path.join(self.storage_dir, 'vecto-resources', '/'.join(resource_name), 'metadata.json')) as f:
            q = load(f)
        self.fetch_file(q['url'])

    def fetch_file(self, url, output_file='tmp'):
        with open(path.join(self.storage_dir, output_file), 'wb') as file:
            response = get(url)
            file.write(response.content)

    def update_directory_structure(self):
        dir_struct = {}
        repo_storage_dir = path.join(self.storage_dir, self.full_resource_path)
        rootdir = repo_storage_dir.rstrip(sep)
        start = rootdir.rfind(sep) + 1
        for filepath, dirs, files in walk(rootdir):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            folders = filepath[start:].split(sep)
            subdir = dict.fromkeys(files)
            parent = reduce(dict.get, folders[:-1], dir_struct)
            if None in subdir.values():
                  parent[folders[-1]] = folders
            else:
                parent[folders[-1]] = subdir
        self.resources = Resources.wrap(dir_struct)

    def get_resources(self):
        return self.resources
