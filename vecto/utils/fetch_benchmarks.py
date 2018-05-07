from git import Repo
from git.exc import GitCommandError
from os import path

path_to_repo = 'https://github.com/vecto-ai/benchmarks.git'


def fetch_benchmarks(path_to_local_dir=path.join('data', 'benchmarks')):
    try:
        Repo.clone_from('https://github.com/vecto-ai/benchmarks.git', path_to_local_dir)
    except GitCommandError:
        raise ValueError('Directory exists')

if __name__ == "__main__":
    fetch_benchmarks()
