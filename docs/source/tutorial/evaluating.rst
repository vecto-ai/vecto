Intrinsic evaluation
====================

Word analogy task
-----------------

One of the de-facto standard intrinsic evaluations for word embeddings is the word analogy task. The dataset known as the Google test set became the de-facto standard for evaluating word embeddings, but it is not balanced and samples only 15 linguistic relations, with 19,544 questions in total. A newer dataset is `BATS <http://www.aclweb.org/anthology/N16-2002>`_: it is considerably larger (98,000 questions) and is balanced: it contains 40 different relations of 4 types (inflections, derivational morphology, lexicographic and encyclopedic semantics) with 50 unique pairs per relation.

Vecto comes with the script to test 6 different methods of solving word analogies. You can run the script from command
line, indicating the path to the config file as the only argument.

.. code:: python

    python3 -m vecto.benchmarks.analogy /path/to/config_analogy.yaml

The configuration file is structured as follows:

.. code:: python

    path_vectors: [
        "/path/to/your/vsm1/"
        "/path/to/your/vsm2/"
       ]

    alpha: 0.6
    # this is the exponent for Sigma values of SVD embeddings

    normalize: true
    # specifies if embeddings should be normalized

    method: LRCos
    # allowed values are 3CosAdd, 3CosAvg, 3CosMul, SimilarToB, SimilalarToAny, PairDistance, LRCos and LRCosF

    exclude: True
    # specifies if question words should be excluded from possible answers

    path_dataset: "/path/to/the/test/dataset"
    # path to dataset. last segment of the path will be interpreted as dataset name

    path_results: "/path/where/to/save/results"
    # Subfolders for datasets and embeddings willl be created automatically

Vecto also support direct call from **run(embeddings, options)** function.
The **options** has the same parameters as that in **yaml** file.
This function returns a dict, which indicate the word analogy results.

For example, the following lines can be used to get word analogy results:

.. code:: python

    path_model = "./test/data/embeddings/text/plain_no_file_header"
    model = vecto.model.load_from_dir(path_model)
    options = {}
    options["path_dataset"] = "./test/data/benchmarks/analogy/"
    options["path_results"] = "/tmp/vecto/analogy"
    options["name_method"] = "3CosAdd"
    vecto.benchmarks.analogy.analogy.run(model, options)

Dataset
~~~~~~~

The BATS dataset can be `downloaded
here <https://my.pcloud.com/publink/show?code=XZOn0J7Z8fzFMt7Tw1mGS6uI1SYfCfTyJQTV>`__.
The script expects the input dataset to be a tab-separated file formatted as follows:

::

    cat cats
    apple apples

In many cases there is more than one correct answer; they are separated with slashes:

::

    father  dad/daddy
    flower  blossom/bloom
    harbor  seaport/haven/harbour

There is a file with a word pairs list for each relation, and these files are grouped into folders by the type of the relation.
You can also make your own test set to use in Vecto, formatted in the same way.

Analogy solving methods
~~~~~~~~~~~~~~~~~~~~~~~

Consider the analogy :math:`a`::math:`a'` :: :math:`b`::math:`b'`
(:math:`a` is to :math:`a'` as :math:`b` is to :math:`b'`). The script
implements 6 analogy solving methods:

Pair-based methods:

`**3CosAdd** <https://www.aclweb.org/anthology/N13-1090>`__:
:math:`b'=argmax_{~d\in{V}}(cos(b',b-a+a'))`, where
:math:`cos(u, v) = \frac{u\cdot{}v}{||u||\cdot{}||v||}`

`**PairDistance** <http://www.aclweb.org/anthology/W14-1618>`__, aka
PairDirection: :math:`b'=argmax_{~d\in{V}}(cos(b'-b,a'-a))`

`**3CosMul** <http://www.aclweb.org/anthology/W14-1618>`__:
:math:`argmax_{b'\in{V}} \frac{cos(b',b) cos(b',a')} {cos(b',a) + \varepsilon}`
:math:`\varepsilon = 0.001` is used to prevent division by zero)

`**SimilarToB** <http://tallinzen.net/media/papers/linzen_2016_repeval.pdf>`__:
returns the vector the most similar to the :math:`b`.

**SimilarToAny**: returns the vector the most similar to any of
:math:`a`, :math:`a'` and :math:`b` vectors.

Set-based methods: (current state-of-the-art)

`**3CosAvg** <https://www.aclweb.org/anthology/C/C16/C16-1332.pdf>`__:
:math:`b'=argmax_{~b'\in{V}}(cos(b',b+\mathit{avg\_offset}))` , where
:math:`\mathit{avg\_offset}=\frac{\sum_{i=0}^m{a_i}}{m} - \frac{\sum_{i=0}^n{b_i}}{n}`

`**LRCos** <https://www.aclweb.org/anthology/C/C16/C16-1332.pdf>`__
:math:`b'=argmax_{~b'\in{V}}(P_{~(b'\in{target\_class)}}*cos(b',b))`

`**LRCosF** <https://www.aclweb.org/anthology/C/C16/C16-1332.pdf>`__: a
version of LRCos that attempts to only take into account the relevant
distributional features.

*Caveat*: Analogy has been shown to be severely misinterpreted as
evaluation task. First of all, `all of the above methods are biased by
distance in the distributional
space <http://www.aclweb.org/anthology/S17-1017>`__: the closer the
target is, the more likely you are to hit it. Therefore high scores on
analogy task indicate basically to what extent the relations encoded by
a given VSM match the relations in the dataset.

Therefore it would be better to not just provide an average score on the
whole task, as it is normally done, but to look at the scores for
different relations, as that may show what exactly the model is doing.
Since everything cannot be close to everything, it is to be expected
that success in one type of relations would come at the expense of
others.

Correlation with human similarity/relatedness judgements
--------------------------------------------------------

One of the first intrinsic evaluation metrics for distributional meaning representations was correlation with human judgements to what extent words are related. Roughly speaking, a good VSM should have tiger and zoo closer in the vector space than tiger and hammer, because tiger and zoo are intuitively more semantically related. There are several datasets with judgements of relatedness and similarity between pairs of words collected from human subjects. See `(Turney 2006) <https://dl.acm.org/ft_gateway.cfm?id=1174523&ftid=389424&dwn=1&CFID=827319269&CFTOKEN=87143883>`_ for the distinction between relatedness and similarity (or relational and attributional similarity).

You can run this type of test in Vecto as follows:

>>> python3 -m vecto.benchmarks.similarity /path/to/config_similarity.yaml

The ``config_similariy.yaml`` file is structured as

::

    path_vector: /path/to/your/vsm1/
    path_dataset: /path/to/the/test/dataset
    normalize: true      # specifies if embeddings should be normalized


Similar to word analogy task, Vecto also support direct call from **run(embeddings, options)** function.
The following lines can be used to get word similarity results:


.. code:: python

    path_model = "./test/data/embeddings/text/plain_with_file_header"
    model = vecto.model.load_from_dir(path_model)
    options = {}
    options["path_dataset"] = "./test/data/benchmarks/similarity/"
    vecto.benchmarks.similarity.similarity.run(model, options)


The similarity/relatedness score file is assumed to have the following tab-separated format:

::

  tiger   cat 7.35
  book    paper   7.46
  computer    keyboard    7.62

You can use any of the many available datasets, including:
 -  `WordSim 353 <http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/>`_ (there is also a version of WordSim353 split into relatedness and similarity subsets)
 -  `MEN <https://staff.fnwi.uva.nl/e.bruni/MEN>`_
 -  `SimLex <https://www.cl.cam.ac.uk/~fh295/simlex.html>`_
 -  `Rare Words <http://www.bigdatalab.ac.cn/benchmark/bm/dd?data=Rare%20Word>`_
 - `Radinsky Mturk data <https://dl.acm.org/citation.cfm?id=1963455>`_

Please refer to the pages of individual datasets for details on how they were collected and references to them. The collection of the above datasets in the same format can also be downloaded `here <https://my.pcloud.com/publink/show?code=XZCeL07ZaEJhoLIaDYz8kuC2B6YMuuYlhMyV>`_.

**Caveat**: while similarity and relatedness tasks remain one of the most popular methods of evaluating word embeddings, they have serious methodological problems. Perhaps the biggest one is the `unreliability of middle judgements <http://www.aclweb.org/anthology/W16-2507>`__: while humans are good at distinguishing clearly related and clearly
unrelated word pairs (e.g. *cat:tiger* vs *cat:malt*), there is no clear reason for rating any of the many semantic relations higher than the other (e.g. which is more related - *cat:tiger* or *cat:whiskers*)? It is thus likely that the human similarity scores reflect some psychological measures like speed of association and prototypicality rather than something purely semantic, and thus a high score on a similarity task should be interpreted accordingly. This would also explain why a high score on similarity or relatedness does not necessarily predict good performance on downstream tasks.

Extrinsic evaluation
====================

The following tasks will soon be available via vecto:

-  POS tagging
-  Named entity recognition
-  Chunking
