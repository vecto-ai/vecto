import abc
from vecto.utils.metadata import WithMetaData


class Benchmark():
    # TODO: define proper interface

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_result(self, embs, path_dataset):
        raise NotImplementedError
