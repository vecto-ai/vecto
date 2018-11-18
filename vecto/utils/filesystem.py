from os import walk, path, chmod, remove, rmdir
from stat import S_IWUSR


def rmtree(top):
    for root, dirs, files in walk(top, topdown=False):
        for name in files:
            filename = path.join(root, name)
            chmod(filename, S_IWUSR)
            remove(filename)
        for name in dirs:
            rmdir(path.join(root, name))
    rmdir(top)