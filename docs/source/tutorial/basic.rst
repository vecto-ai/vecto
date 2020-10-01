Introduction to Vecto
======================

.. currentmodule:: vecto

This is the tutorial for Vecto. It describes:

* What it is, and why we are developing it.
* what you can do with Vecto.
* the roadmap of the project.

Both the library and the documentation are actively developed, check back for more! If you have questions, or would like to contribute, feel free to get in touch on `github <https://github.com/undertherain/Vecto>`_.

What is Vecto?
-------------------

Vecto is an open-source Python library for working with vector space models (VSMs), including various word embeddings such as word2vec. Vecto can load various popular formats of VSMs and retrieve nearest neighbors of a given vector. It includes a growing list of benchmarks with which VSMs are evaluated in most current research, and a few visualization tools. It also includes a growing list of modules for creating VSMs, both explicit and based on neural networks. 

Why do you bother?
--------------------

There are a few other libraries for working with VSMs, including gensim and spacy. Vecto differs from them in that its primary goal is to facilitate principled, systematic research in providing **a framework for reproducible experiments** on VSMs.

From the academic perspective, this matters because this is the only way to understand more about what VSMs are and what kind of meaning representation they offer.

From the practical perspective, this matters because otherwise we can not tell which VSM would be the best to use for what task. Existing extrinsic evaluations of VSMs such as popular word similarity, relatedness, analogy and intrusion tasks have methodological problems and do not correlate well with performance on all extrinsic tasks. Therefore basically to pick the best representation for a task you have to try different kinds of VSMs until you find the best-performing one.

Furthermore, there is the important and unpleasant part of parameter tuning and optimizing for a particular task. `Levy et al. (2015) <http://www.aclweb.org/anthology/Q15-1016>`_ showed that the choice of hyperparameters may make more of a difference than the choice of model itself. Even more frustratingly, when you have a relatively comprehensive task covering a wide range of linguistic relations, you may find that the parameters beneficial to a part of the task are detrimental for another part `(Gladkova et al. 2016) <http://www.aclweb.org/anthology/N16-2002>`_.

The neural parts of Vecto is implemented in `Chainer <https://www.chainer.org>`_, a new deep learning framework that is friendly to high-performance multi-GPU environments. This should make Vecto useful in both academic and industrial settings.
