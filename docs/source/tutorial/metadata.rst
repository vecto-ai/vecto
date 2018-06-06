The metadata
------------

.. currentmodule:: vsmlib


Vecto attempts to record and track as much information as possible about each embedding and each experiment you run. All the information about VSMs is stored in a `metadata.json` file in the same folder as the VSM itself.

Vecto can be used to work with VSMs that were trained elsewhere and may not come with any metadata. However, even in this case, we encourage the users to try and find out and record as much of the metadata as possible,as soon as possible. We have all been in the situation where, long after you have published a paper and forgotten all about that project, you need to reuse some code or repeat an experiment - and that it's nigh impossible, because the code is unreadabe, filenames are criptic, and filepaths are long gone.

Moreover, keeping track of the metadata is also something that would force the researchers to be more aware of all these different hidden variables in their experiments. That would (1) prevent them from misinterpreting the properties of their models, and (2) provide some ideas about what could be tweaked. 

The corpus metadata
~~~~~~~~~~~~~~~~~~~

It all starts with the corpus. Actually, as many corpora as you like, since it is common practice to combine corpora to train a model (to increase the volume of data, to diversify it, or in fancy curriculum learning). Here is a sample metadata file you can use as a template to describe your corpus.

Vecto records the following metadata:

:todo: a page about domains

.. glossary::

   environment
      A structure where information about all documents under the root is
      saved, and used for cross-referencing.  The environment is pickled
      after the parsing stage, so that successive runs only need to read
      and parse new and changed documents.

   source directory
      The directory which, including its subdirectories, contains all
      source files for one Sphinx project.

.. glossary::

   id
      An identifier of the corpus, unique in the collection.
   size
      The size of the corpus (in tokens).
   name
      The (preferably short) name of the corpus, often used to identify the models built from it.
   description
      The freeform description of the corpus, such as the domains it covers.
   source
      Where the corpus was obtained.
   domain
      The list of the domains of the texts, such as **news**, **encyclopedia**, **fiction**, **medical**, **spoken**, or **web**. If the corpus covers only one domain, the list only contains one item; otherwise several can be listed. We suggest using **general** only for balanced, representative corpora such as `BNC <http://www.natcorp.ox.ac.uk/corpus/creating.xml>`_ that make a conscious effort to represent different registers.
   language
      A list containing the language codes for the corpus. There will be just one entry in case of monolingual corpora (e.g. _["en"]_), and for parallel or multilingual corpora there will be several (_["en", "de"]_).
   encoding
      The encoding of the corpus files.
   format
      The format of the corpus. Some frequent options include: one-corpus-per-line, one-sentence-per-line, one-paragraph-per-line, one-word-per-line, vertical-format
   date
      The date when the corpus (or its text source) was published. It can be the date of a Wikipedia dump (e.g. _2018-07_), or the year when the paper presenting the resource came out (e.g. _2017_).
   path
      The path to the local copy of the corpus files.
   cite
      The bibtex entry for the paper presenting the resource, that should be referenced in subsequent work building on or using the resource. It should be bibtex rather than biblatex, as most NLP publishers have not made the switch yet.
   pre-processing
      The pre-processing steps used in preparing this resource, described in freeform text.
   cleanup
      Markup removal, format conversion, encoding, de-duplication (freeform description, URL or path to the pre-processing script)
   lowercasing
      **True** if the corpus was lowercased, **False** otherwise.
   tokenization
      The tokenizer that was used, if any (URL or path to the script, name, version).
   lemmatization
      The lemmatizer that was used, if any (URL or path to the script, name, version).
   stemming
      The stemmer that was used, if any (URL or path to the script, name, version).
   POS_tagging
      The POS-tagger that was used, if any (URL or path to the script, name, version).
   syntactic_parsing
      The syntactic parser that was used, if any (URL or path to the script, name, version).
   Semantic_parsing
      The semantic parser that was used, if any (URL or path to the script, name, version).
   Other_preprocessing
      Any other pre-processing that was performed, if any (URL or path to the script, name, version).

:todo: the format section should link to the input of embedding models


.. code-block:: json

    {
    "corpus_01":   {
                    "id": "",
                    "size": ,
                    "name": "",
                    "description": "",
                    "source": "",
                    "domain": "",
                    "language": ["english"],
                    "encoding": "",
                    "format": "",
                    "date": "",
                    "path": "",

                    "pre-processing": {
                                    "cleanup": "",
                                    "lowercasing": ,
                                    "tokenization": "",
                                    "lemmatization": "",
                                    "stemming": "",
                                    "POS_tagging": "",
                                    "syntactic_parsing": "",
                                    "semantic_parsing": "",
                                    "other_preprocessing": "",
                                    }
                    }
    "corpus_02":   {
                    "id": "",
                    "size": ,
                    "name": "",
                    "description": "",
                    "source": "",
                    "domain": "",
                    "language": ["english"],
                    "encoding": "",
                    "format": "",
                    "date": "",
                    "path": "",

                    "pre-processing": {
                                    "cleanup": "",
                                    "lowercasing": ,
                                    "tokenization": "",
                                    "lemmatization": "",
                                    "stemming": "",
                                    "POS_tagging": "",
                                    "syntactic_parsing": "",
                                    "semantic_parsing": "",
                                    "other_preprocessing": ""
                                    }
                    }
    }

The vocab metadata
~~~~~~~~~~~~~~~~~~

The vocab files are basically lists of the vocabulary of word embeddings. Sometimes they are stored separately from the numerical data as plain-text, one-word-per-line files (e.g. when the numerical data itself is stored in .npy format). Vecto expects such files to have a ".vocab" extension.

The vocab files can have associated metadata as follows.

.. glossary::

   size: The number of token types.
   min_frequency: The minimum frequency cut-off point.
   timestamp: When the vocab file was produced
   filtering: A freeform description of any filtering applied to the vocabulary, if any.
   lib_version: The version of Vecto with which a given vocab file was produced (generated automatically by Vecto).
   system_info: The system in which the vocab file was produced (generated automatically by Vecto).
   timing: todo
   source_corpus: Includes the corpus metadata, as described in `The corpus metadata`_ section.

:todo: link to the vocab filtering section, if any
:todo: explain timing

.. code-block:: json

    {
    "original": {
                "size": ,
                "min_frequency": ,
                "timestamp": "",
                "lib_version": "",
                "system_info": "",
                "timing": "",
                "source_corpus": {
                                }
                }
    "filtered": {
                "size": ,
                "timestamp": "",
                "lib_version": "",
                "system_info": "",
                "timing": ""
                }
    }


The embeddings metadata
~~~~~~~~~~~~~~~~~~~~~~~

The metadata collected in training of embeddings is hard to standartize, because essentially it needs to describe all the parameters of a given model, and they differ across models. Therefore this section only provides a sample, and the full list of parameters (which correspond to metadata) can be found in descriptions of the implementations of different models in the library.

:todo: link to the library of embeddings

Some of the frequent parameters applicable to most-if-not-all models include:

.. glossary::

   model: The name of the model, such as CBOW or GloVe.
   window: The window size
   dimensionality: The number of vector dimensions.
   context: The type of context, as described by `Li et al <http://www.aclweb.org/anthology/D17-1257>`_. Four common combinations are **linear_unbound** (the bag-of-words symmetrical context, the most commonly used), **linear_bound** (linear context that takes word order into account), **deps_unbound** (the dependency-based context which takes into account all words in a syntactic relation to the target word), and **deps_boun** (a version of the latter which differentiates between different syntactic relations). See the paper for mor details.
   epochs: The number of epochs for which the model was trained.
   cite: The bibtex entry for the paper presenting the resource, that should be referenced in subsequent work building on or using the resource. It should be bibtex rather than biblatex, as most NLP publishers have not made the switch yet.
   vocabulary: The vocabulary metadata as described in `The vocab metadata`_, which also includes the corpus metadata.

.. code-block:: json

    {
        "model": "",
        "window": ,
        "dimensionality": ,
        "context": "",
        "epochs": ,
        "cite": "",
        "vocabulary": {
                    }

        "lib_version": "",
        "system_info": "",
    }

:todo: what to do with the lib version and system info?

The datasets metadata
~~~~~~~~~~~~~~~~~~~~~

The task datasets should be accompanied by the following metadata:

.. glossary::

   task: The task for which the dataset is applicable, such as **word_analogy** or **word_relatedness**.
   language: A list containing the language codes for the corpus. There will be just one entry in case of monolingual corpora (e.g. _["en"]_), and for parallel or multilingual corpora there will be several (_["en", "de"]_).
   name: The (preferably short) name of the dataset, such as **WordSim353**.
   description: The freeform brief description of the dataset, preferably including anything special about this dataset that distinguishes it from other datasets for the same task.
   domain: The domain of the dataset, such as **news**, **encyclopedia**, **fiction**, **medical**, **spoken**, or **web**. We suggest using **general** only for datasets that do not target any particular domain.
   date: The date the resource was published.
   source: The source of the resource (e.g. a modification of another dataset, or something created by the authors from scratch or on the basis of some data that was not previously used as a dataset for the same task).
   version: The version of the dataset (useful when you are developing one).
   Size: The size of the dataset. The units depend on the task: it can be e.g. **353 pairs** for a similarity or analogy dataset.
   cite: The bibtex entry for the paper presenting the resource, that should be referenced in subsequent work building on or using the resource. It should be bibtex rather than biblatex, as most NLP publishers have not made the switch yet.

.. code-block:: json

    {
        "task": "",
        "language": ["english"],
        "name": "",
        "description": "",
        "domain": "",
        "date": "",
        "source": "",
        "version": "",
        "size": "",
        "cite": ""
    }

:todo: ids? paths?

The experiment metadata
~~~~~~~~~~~~~~~~~~~~~~~

As with the training of embeddings, different experiments involve different sets of metadata. The parameters of each model included in the Vecto library is described in the corresponding library page. In addition to that, the metadata for each experiment will automatically include the metadata for the dataset and embeddings (which also includes the corpus metadata).

Some of the generic metadata fields that are applicable to all experiments include:

.. glossary::

   name: The (hopefully descriptive) name of the model, such as **LogisticRegression**.
   task: The type of the task that this model is applicable to (e.g. **word_analogy** or **text_classification**).
   description: A brief description of the implementation, preferably including its use case (e.g. a sample implementation in a some framework, a standard baseline for some task, a state-of-the-art model.)
   author: The author of the code (for unpublished models).
   implementation: The id of the implementation in the Vecto library, in case there are several alternative implementations for the same task.
   framework: Machine learning library that this implementation uses, such as **scikit-learn**, **Chainer** or **Keras** (if any).
   version: The version of the implementation, if any.
   date: The date when the code was published or contributed.
   source: If the code is reimplementation of something else, this is the field to indicate it.
   cite: The bibtex entry for the paper presenting the code, that should be referenced in subsequent work building on or comparing with this implementation. It should be bibtex rather than biblatex, as most NLP publishers have not made the switch yet.

.. code-block:: json

    {
        "name": "",
        "task": "",
        "description": "",
        "author": "",
        "implementation": "",
        "framework": "",
        "version": "",
        "date": "",
        "source": "",
        "cite": ""
    }


Accessing the metadata in Vecto
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All metadata is accessible from vsmlib while you are experimenting on dozens of VSMs you have built, facilitating both parameter search for a particular task and observations on what properties of VSMs result in what aspects of their performance.

You can access the VSM metadata as follows: 

The name of the model, which is the name directory in which it is stored. For models generated with VSMlib, interpretable folder names with parameters are generated automatically. 

>>> print(my_vsm.name)
w2v_comb2_w8_n25_i6_d300_skip_300


You can also access the metadata as a Python dictionary:

>>> print(my_vsm.metadata)
{'size_dimensions': 300, 'dimensions': 300, 'size_window': '8'}
