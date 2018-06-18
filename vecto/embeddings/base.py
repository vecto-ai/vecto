import abc
from vecto.utils.metadata import WithMetaData


class WordEmbeddings(WithMetaData, metaclass=abc.ABCMeta):
    # TODO: define proper interface

    @abc.abstractmethod
    def get_vector(self, w):
        pass
