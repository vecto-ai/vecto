import abc
from vecto.utils.metadata import WithMetaData


class Benchmark(WithMetaData):
    # TODO: define proper interface

    @abc.abstractmethod
    def get_result(self):
        pass
