import abc
from vecto.utils.metadata import WithMetaData


class WordEmbeddings(WithMetaData):
    # TODO: define proper interface

    @abc.abstractmethod
    def get_vector(w):
        pass

    @property
    @abc.abstractmethod
    def dimensions_number(self):
        pass
