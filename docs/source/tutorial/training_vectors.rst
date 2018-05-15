Training new models
===================

.. currentmodule:: vsmlib


This page describes how to train vectors with the models that are currently implemented in VSMlib.


Word2vec
--------

`Word2vec <https://arxiv.org/pdf/1301.3781.pdf>`_ is arguably the most popular word embedding model. 
We provide implementation of extended word2vec model, which can be trained on linear and dependency-based contexts,
with bound and unbound context representations.

Additionally we provide an implementation which considers characters rather than words to be the minimal units. This enables it to take advantage of morphological information: as far as a word-level models such as word2vec is concerned, "walk" and "walking" are completely unrelated, except  through similarities in their distributions.

To train word2vec embeddings vsmlib can be envoked via the command line interface:

>>> python3 -m vsmlib.embeddings.train_word2vec

The command line parameters are as 

--dimensions                  size of embeddings
--context_type                context type [linear' or 'deps'], for deps context, the annotated corpus is required
--context_representation      context representation ['bound' or 'unbound']
--window                      window size')
--model                       base model type ['skipgram' or 'cbow']
--negative-size               number of negative samples
--out_type                    output model type ["hsm": hierarchical softmax, "ns": negative sampling, "original": no approximation]
--subword                     specify if subword-level approach should be used ["none", "rnn"]
--batchsize                   learning minibatch size
--gpu                         GPU ID (negative value indicates CPU)
--epochs                      number of epochs to learn
--maxWordLength               max word length (only used for char-level subword)
--path_vocab                  path to the vocabulary
--path_corpus                 path to the corpus
--path_out                    path to save embeddings
--test                        run in test mode
--verbose                     verbose mode


Alternatively, word2vec training can be done though vsmlib python API. 

>>> vsmlib.embeddings.train_word2vec.train(args)

The arguments are argparse.namespace identical to command line arguments. Instance of ModelDense is returned. 

Realted papers: original w2v, Bofang, Mnih, subword.

::

 @inproceedings{MikolovChenEtAl_2013_Efficient_estimation_of_word_representations_in_vector_space,
  title = {Efficient Estimation of Word Representations in Vector Space},
  urldate = {2015-12-03},
  booktitle = {Proceedings of International Conference on Learning Representations (ICLR)},
  author = {Mikolov, Tomas and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  year = {2013}}

::

 @inproceedings{Li2017InvestigatingDS,
  title={Investigating Different Syntactic Context Types and Context Representations for Learning Word Embeddings},
  author={Bofang Li and Tao Liu and Zhe Zhao and Buzhou Tang and Aleksandr Drozd and Anna Rogers and Xiaoyong Du},
  booktitle={EMNLP},
  year={2017}}


