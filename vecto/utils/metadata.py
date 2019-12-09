import os
from .data import load_json, save_json
from vecto._version import VERSION

METADATA_SUFFIX = 'metadata.json'


def make_metadata_path(fname):
    if os.path.isdir(fname):
        return os.path.join(fname, METADATA_SUFFIX)
    return '{}.{}'.format(fname, METADATA_SUFFIX)


def save_metadata(data, base_path):
    save_json(data, make_metadata_path(base_path))


def try_load_metadata(base_path):
    try:
        return load_json(make_metadata_path(base_path))
    except IOError:
        return {}


def get_full_typename(obj):
    # cls = type(obj)
    if obj.__class__.__name__ == 'function':
        clsname = obj.__name__
    else:
        clsname = obj.__class__.__name__
    return '{}.{}'.format(obj.__module__, clsname)


class WithMetaData(object):
    """
    Base object for all objects with metadata. Contains utilities for metadata loading from files, storing to files,
    collecting/merging etc.

    User of this class is responsible for calling __init__ or init_metadata and save_metadata
    in proper places of inheritor.
    """

    def __init__(self, base_path=None, **other_metadata):
        """
        see init_metadata
        """
        self.metadata = {}
        self.init_metadata(base_path=base_path, **other_metadata)

    def init_metadata(self, base_path=None, **other_metadata):
        """
        :param base_path: path from which metadata.json path will be constructed
        :param other_metadata: anything json serializable
        """
        # self._metadata = {"vecto_version": VERSION}
        if base_path is not None:
            self.metadata['_base_path'] = base_path
            self.load_metadata(base_path)
        self.metadata.update(other_metadata)
        self.metadata['_class'] = get_full_typename(self)

    def save_metadata(self, base_path):
        """
        :param base_path: path from which metadata.json path will be constructed
        """
        save_metadata(self.metadata, base_path)

    def load_metadata(self, base_path):
        """
        :param base_path: path from which metadata.json path will be constructed
        """
        self.metadata.update(try_load_metadata(base_path))
