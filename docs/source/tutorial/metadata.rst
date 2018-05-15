The metadata
------------

.. currentmodule:: vsmlib


VSMlib attempts to record and track as much information as possible about each VSM and each experiment you run. All the information about VSMs is stored in a `metadata.json` file in the same folder as the VSM itself. 

VSMlib can also be used to work with VSMs that were trained elsewhere and may not come with any metadata. Even in this case, we encourage the users to try and find out and record as much of the metadata as possible,as soon as possible. We have all been in the situation where, long after you have published a paper and forgotten all about that project, you need to reuse some code or repeat an experiment - and that it's nigh impossible, because the code is unreadabe, filenames are criptic, and filepaths are long gone.

Moreover, keeping track of the metadata is also something that would force the researchers to be more aware of all these different hidden variables in their experiments. That would (1) prevent them from misinterpreting the properties of their models, and (2) provide some ideas about what could be tweaked. 

The collected metadata depends on the particular VSM and/or experiment you are running. For VSMs, this includes such parameters as the source corpus, window and vector size, normalization, number of training epochs, minimal frequency, subsampling, size of vocabulary, etc.  

All metadata is accessible from vsmlib while you are experimenting on dozens of VSMs you have built, facilitating both parameter search for a particular task and observations on what properties of VSMs result in what aspects of their performance.

You can access the VSM metadata as follows: 

The name of the model, which is the name directory in which it is stored. For models generated with VSMlib, interpretable folder names with parameters are generated automatically. 

>>> print(my_vsm.name)
w2v_comb2_w8_n25_i6_d300_skip_300


You can also access the metadata as a Python dictionary:

>>> print(my_vsm.metadata)
{'size_dimensions': 300, 'dimensions': 300, 'size_window': '8'}
