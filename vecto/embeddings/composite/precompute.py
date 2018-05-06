import numpy as np
from vecto.embeddings.dense import WordEmbeddingsDense
from vecto.vocabulary import create_from_iterator
from vecto.corpus.tokenization import word_tokenize_txt


def precompute_composite_embeddings(composition, objects, items_getter=word_tokenize_txt):
    """
    Build dense embeddings by computing vectors for all compound `objects`.
    :param composition: any implementor of WordEmbeddings interface, which is able to build vectors for compound objects.
    :param objects: a list of hashable objects (such as tuples of tokens)
    :param items_getter: a function which converts each object to an iterable
    :return: WordEmbeddingsDense
    """
    vocab = create_from_iterator(objects)
    matrix = np.array([composition.get_vector(items_getter(obj)) for obj in vocab.lst_words])
    return WordEmbeddingsDense(matrix=matrix, vocab=vocab)
