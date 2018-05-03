import abc

from vecto.embeddings.base import WordEmbeddings
from vecto.utils.blas import make_normalizer


class BaseComposition(WordEmbeddings):
    def __init__(self, base_embeddings, norm='unnormed'):
        super(BaseComposition, self).__init__(base_embeddings=base_embeddings.metadata,
                                              norm=norm)
        self._base_embeddings = base_embeddings
        self._normalizer = make_normalizer(norm)

    def get_vector(self, obj):
        element_vectors = self._get_element_vectors(obj)
        obj_vector = self._aggregate_vectors(element_vectors)
        return self._normalizer(obj_vector)

    @property
    def dimensions_number(self):
        return self._base_embeddings.dimensions_number

    def _get_element_vectors(self, obj):
        """
        Yields vector for each element of a compound `obj` object.
        Yields results in a lazy fashion to save memory.
        :param obj: A compound object (like tokenized sentence).
        :return: An iterator of vectors with their weights
        """
        for w in obj:
            if self._base_embeddings.has_word(w):
                yield self._base_embeddings.get_vector(w), 1.0

    @abc.abstractmethod
    def _aggregate_vectors(self, vectors_with_weights):
        pass
