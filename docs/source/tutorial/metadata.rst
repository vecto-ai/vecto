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

.. glossary::

   size : The size of the corpus (in tokens).
   name : The (preferably short) name of the corpus, often used to identify the models built from it.
   description: The freeform description of the corpus, such as the domains it covers.
   source: Where the corpus was obtained.
   language: A list containing the language codes for the corpus. There will be just one entry in case of monolingual corpora (e.g. _["en"]_), and for parallel or multilingual corpora there will be several (_["en", "de"]_).
   encoding: The encoding of the corpus files.
   format: The format of the corpus. Some frequent options include: one-corpus-per-line, one-sentence-per-line, one-paragraph-per-line, one-word-per-line, vertical-format
   date: The date when the corpus (or its text source) was published. It can be the date of a Wikipedia dump (e.g. _2018-07_), or the year when the paper presenting the resource came out (e.g. _2017_).
   cite: The bibtex entry for the paper presenting the resource, that should be referenced in subsequent work building on or using the resource. It should be bibtex rather than biblatex, as most NLP publishers have not made the switch yet.
   pre-processing: The pre-processing steps used in preparing this resource, described in freeform text.
   cleanup: Markup removal, format conversion, encoding, de-duplication (freeform description, URL or path to the pre-processing script)
   tokenization: The tokenizer that was used, if any (URL or path to the script, name, version).
   lemmatization: The lemmatizer that was used, if any (URL or path to the script, name, version).
   stemming: The stemmer that was used, if any (URL or path to the script, name, version).
   POS_tagging: The POS-tagger that was used, if any (URL or path to the script, name, version).
   syntactic_parsing: The syntactic parser that was used, if any (URL or path to the script, name, version).
   Semantic_parsing: The semantic parser that was used, if any (URL or path to the script, name, version).
   Other_preprocessing: Any other pre-processing that was performed, if any (URL or path to the script, name, version).

:todo: the format section should link to the input of embedding models


.. code-block:: json

    {
    "corpus_01":   {
                    "size": ,
                    "name": "",
                    "description": "",
                    "source": "",
                    "language": ["eng"],
                    "encoding": "",
                    "format": "",
                    "date": "",

                    "pre-processing": {
                                    "cleanup": "",
                                    "tokenization": "",
                                    "lemmatization": "",
                                    "stemming": "",
                                    "POS_tagging": "",
                                    "syntactic_parsing": "",
                                    "semantic_parsing": "",
                                    "other_preprocessing": "",
                                    }
                    }
    }

The vocab metadata
~~~~~~~~~~~~~~~~~~

The embeddings metadata
~~~~~~~~~~~~~~~~~~~~~~~

The collected metadata depends on the particular VSM and/or experiment you are running. For VSMs, this includes such parameters as the source corpus, window and vector size, normalization, number of training epochs, minimal frequency, subsampling, size of vocabulary, etc.

The experiment metadata
~~~~~~~~~~~~~~~~~~~~~~~




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
