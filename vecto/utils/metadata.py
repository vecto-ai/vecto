import os
from .data import load_json, save_json


METADATA_SUFFIX = 'metainfo.json'


def make_metadata_path(fname):
    if os.path.isdir(fname):
        return os.path.join(fname, METADATA_SUFFIX)
    else:
        return '{}.{}'.format(fname, METADATA_SUFFIX)


def save_metadata(data, base_path):
    save_json(data, make_metadata_path(base_path))


def try_load_metadata(base_path, default={}):
    try:
        return load_json(make_metadata_path(base_path))
    except IOError:
        return default


class WithMetaData(object):
    def __init__(self, base_path=None, **other_metadata):
        self._metadata = {}
        if base_path is not None:
            self.metadata['base_path'] = base_path
            self.load_metadata(base_path)
        self.metadata.update(other_metadata)

    def save_metadata(self, base_path):
        save_metadata(self.metadata, base_path)

    def load_metadata(self, base_path):
        self.metadata.update(try_load_metadata(base_path))

    @property
    def metadata(self):
        return self._metadata
