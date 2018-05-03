import numpy as np
from vecto.embeddings.dense import WordEmbeddingsDense
from vecto.vocabulary import create_from_iterator


def precompute_composite_embeddings(composition, objects):
    """
    Build dense embeddings by computing vectors for all compound `objects`.
    :param composition: any implementor of WordEmbeddings interface, which is able to build vectors for compound objects.
    :param objects: a list of hashable objects (such as tuples of tokens)
    :return: WordEmbeddingsDense
    """
    vocab = create_from_iterator(objects)
    matrix = np.array([composition.get_vector(obj) for obj in vocab.lst_words])
    return WordEmbeddingsDense(matrix=matrix, vocab=vocab)
