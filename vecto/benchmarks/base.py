import abc
from vecto.utils.metadata import WithMetaData


class Benchmark():
    # TODO: define proper interface

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_result(self, embeddings, path_dataset):
        raise NotImplementedError

    @abc.abstractmethod
    def read_test_set(self, path):
        raise NotImplementedError