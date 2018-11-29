Basic operations
=====================

.. currentmodule:: vecto

Supported VSM formats
-----------------------

At the moment the following data formats are supported:

*    .bin format of word2vec (the file has to be called "vectors.bin")
*    .npy arrays with separate vocab files
*    .txt plain-text vectors
*    sparse vectors in hp5 format

:todo: fasttext .vec format?

Importing vectors
-------------------

Vecto assumes a one-folder-per-vsm folder structure. All files related to the same vsm - the metadata, vectors, vocab files,  etc. - must all be stored in one directory. If the vector files has the correct extension (.npy, .txt, .bin, .hp5), the library will attempt to "guess" the correct module to load it with.

>>> import vecto
>>> path_to_vsm = "/path/to/your/model"
>>> my_vsm = vecto.model.load_from_dir(path_to_vsm)

The name of the model is the name directory in which the vector files are stored. For models generated with Vecto, interpretable folder names with parameters are generated automatically.

>>> print(my_vsm.name)
w2v_comb2_w8_n25_i6_d300_skip_300

You can access the VSM metadata (recorded in metadata.json file located in the same directory as the VSM) as a Python dictionary:

>>> print(my_vsm.metadata)
{'size_dimensions': 300, 'dimensions': 300, 'size_window': '8'}

Getting top similar neighbors of a word
---------------------------------------

>>> my_vsm.get_most_similar_words("apple", cnt=5)
[['apple', 1.0000000999898755],
 ['fruit', 0.61400752577032369],
 ['banana', 0.58657183882050712],
 ['plum', 0.5850951585421692],
 ['apples', 0.58464719369713347]]

This method takes an optional ``cnt`` argument specifying how many top similar neighbors to output (the default is 10). Note that the top similar vector is always the target word itself.

If you need to compute nearest neighbors for many words, this function works
faster if the VSM is normalized. If it was generated with vecto, the
normalization will be recorded in metadata, and can be checked with `:meth:
.normalized()` method. Vecto will automatically check for normalization and use
the faster routine if possible. If not, you can first normalize your model as
follows:

>>> my_embeddings.normalize()

Please note that this will consume additional memory.

If you're going to use the same normalized model several times, you can
avoid re-doing the normalization with:

>>> my_embeddings.cache_normalized_copy()

`.get_most_similar_vectors()` enables you to do the same as ``.get_most_similar_words()``, but searching the top neighbors by the vector representation rather than its label.

Note:

  The speed of vector neighborhood computation depends on whether your numpy
  package has access
  to the right linear algebra library - MKL, OpenBLAS or whatever is available
  for your system. With the OpenBLAS and 4 Ghz Core i7-6700K processor in Ubuntu we're
  processing 900 words for 300K 500-dimensional embeddings in under three
  minutes.

  If you do have the library, but the neighbor extraction is   still slow,
  check if it is actually used by numpy. This can be done as
  follows:

  >>> import numpy as np
  >>> np.show_config()



Words to vectors and back
-------------------------

First, you need to import your model from a directory that holds only that model (.npy, .bin, .hp5 or .txt formats) and any associated files.

getting the vector representation of a word

>>> my_vsm.get_row("apple")
array([-0.17980662,  0.27027196, -0.33250481,  ... -0.22577444], dtype=float32)

You can use the above top-similar function to get the label of the vector most corresponding to your vector in your VSM vocabulary:

>>> vsm.get_most_similar_vectors(vsm.get_row("apple"))

Filtering the vocabulary of a VSM
---------------------------------

In certain cases it may be useful to filter the vocabulary of a pre-trained VSM, e.g. to ensure that two models you are comparing have the same vocabulary. Vecto provides a ``.filter_by_vocab()`` method that returns a new model instance, the vocabulary of which contains only the words in the provided Python list of words. The list can be empty.

>>> my_vsm.get_most_similar_words("cat", cnt=5)
[['cat', 1.0],
 ['monkey', 0.95726192],
 ['dog', 0.95372206],
 ['koala', 0.94773519],
 ['puppy', 0.94360757]]
>>> my_new_vsm = my_vsm.filter_by_vocab(["dog", "hotdog", "zoo", "hammer", "cat"])
>>> my_new_vsm.get_most_similar_words("cat", cnt=5)
[['cat', 1.0],
 ['dog', 0.95372206],
 ['hotdog', 0.84262532],
 ['hammer', 0.80627602],
 ['zoo', 0.7463485]]
