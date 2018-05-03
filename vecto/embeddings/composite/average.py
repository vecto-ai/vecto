import numpy as np

from .base import BaseComposition


class IDFWeightedComposition(BaseComposition):
    def __init__(self, base_embeddings, vocab=None, **kwargs):
        super().__init__(base_embeddings, **kwargs)
        if vocab is not None:
            self.metadata['vocab'] = vocab.metadata
            self._vocab = vocab
        else:
            self._vocab = self._base_embeddings.vocabulary

    def _get_element_vectors(self, obj):
        for w in obj:
            if self._base_embeddings.has_word(w):
                idf = 1 / (1 + self._vocab.get_frequency(w))
                yield self._base_embeddings.get_vector(w), idf


class ArithmeticMeanVectorMixin(object):
    def _aggregate_vectors(self, vectors_with_weights):
        result = np.zeros(self.dimensions_number, dtype='float32')
        norm = 0
        for vec, weight in vectors_with_weights:
            result += vec * weight
            norm += weight
        return (result / norm) if norm > 0 else result


class ArithmeticMeanVector(ArithmeticMeanVectorMixin, BaseComposition):
    pass


class IDFArithmeticMeanVector(ArithmeticMeanVectorMixin, IDFWeightedComposition):
    pass


class GeometricMeanVectorMixin(object):
    def _aggregate_vectors(self, vectors_with_weights):
        result = np.ones(self.dimensions_number, dtype='float32')
        norm = 0
        for vec, weight in vectors_with_weights:
            result *= vec ** weight
            norm += weight
        return (result ** (1 / norm)) if norm > 0 else result


class GeometricMeanVector(GeometricMeanVectorMixin, BaseComposition):
    pass


class IDFGeometricMeanVector(GeometricMeanVector, IDFWeightedComposition):
    pass
