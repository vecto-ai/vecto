#!/usr/bin/env python
"""Sample script of recurrent neural network language model.
This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm
"""
from __future__ import division
# import argparse

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import os
import datetime
from scipy.stats.stats import spearmanr
import os
import math
from ..base import Benchmark
import csv
import tempfile
import os
from sklearn.neural_network import MLPClassifier
import subprocess
from sklearn.linear_model import LogisticRegression
import numpy as np


# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):

    def get_embed_from_wv(self, w):
        return self.wv[w]

    def __init__(self, n_vocab, n_units, model_name, wv, window_size):
        super(RNNForLM, self).__init__()
        with self.init_scope():
            # self.embed = L.EmbedID(n_vocab, n_units)
            self.embed = self.get_embed_from_wv
            self.model_name = model_name
            self.wv = wv

            if self.model_name == 'lr':
                self.lr = L.Linear(n_units * window_size, n_vocab)

            if self.model_name == '2FFNN':
                self.nn1 = L.Linear(n_units * window_size, n_units * window_size)
                self.nn2 = L.Linear(n_units * window_size, n_vocab)

            if self.model_name == 'rnn' or self.model_name == 'lstm':
                self.l3 = L.Linear(n_units, n_vocab)
            if self.model_name == 'rnn':
                self.l1 = L.LSTM(n_units, n_units)
                self.l2 = L.LSTM(n_units, n_units)
            if self.model_name == 'lstm':
                self.l1 = L.LSTM(n_units, n_units)
                self.l2 = L.LSTM(n_units, n_units)
            self.window_size = window_size

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        if self.model_name == 'rnn' or self.model_name == 'lstm':
            self.l1.reset_state()
            self.l2.reset_state()

    def __call__(self, x):
        if self.model_name == 'rnn' or self.model_name == 'lstm':
            h0 = self.embed(x[:, self.window_size - 1])
            h1 = self.l1(h0)
            # h2 = self.l2(F.dropout(h1))
            y = self.l3(h1)
        if self.model_name == 'lr' or self.model_name == '2FFNN':
            h = self.embed(x)
            h = h.reshape((h.shape[0], h.shape[1] * h.shape[2]))
        if self.model_name == 'lr':
            y = self.lr(h)
        if self.model_name == '2FFNN':
            y = self.nn1(h)
            y = F.tanh(y)
            y = self.nn2(y)
        return y


# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, window_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0
        # use -1 instead of None internally
        self._previous_epoch_detail = -1.
        self.window_size = window_size

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration

        real_iteration = self.iteration

        cur_words_list = []
        for i in range(self.window_size):
            cur_words = self.get_words()
            cur_words_list.append(cur_words)
            self._previous_epoch_detail = self.epoch_detail
            self.iteration += 1
        cur_words_list = np.asarray(cur_words_list)
        cur_words_list = cur_words_list.transpose()
        # print(cur_words_list.shape)
        next_words = self.get_words()

        self.iteration = real_iteration + 1
        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words_list, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    # @property
    # def previous_epoch_detail(self):
    #     if self._previous_epoch_detail < 0:
    #         return None
    #     return self._previous_epoch_detail

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    # def serialize(self, serializer):
    #     # It is important to serialize the state to be recovered on resume.
    #     self.iteration = serializer('iteration', self.iteration)
    #     self.epoch = serializer('epoch', self.epoch)
    #     try:
    #         self._previous_epoch_detail = serializer(
    #             'previous_epoch_detail', self._previous_epoch_detail)
    #     except KeyError:
    #         # guess previous_epoch_detail for older version
    #         self._previous_epoch_detail = self.epoch + \
    #                                       (self.current_position - self.batch_size) / len(self.dataset)
    #         if self.epoch_detail > 0:
    #             self._previous_epoch_detail = max(
    #                 self._previous_epoch_detail, 0.)
    #         else:
    #             self._previous_epoch_detail = -1.


# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)

            # Compute the loss at this time step and accumulate it
            # loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))
            loss += optimizer.target(x, t)

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


class Language_modeling(Benchmark):

    def __init__(self, normalize=True, window_size=5, method='lstm', test=True):  # 'lr', '2FFNN', 'lstm'
        self.normalize = normalize
        self.window_size = window_size
        self.method = method

        self.batchsize = 1000
        self.bproplen = 35
        self.epoch = 10
        self.gpu = -1
        self.gradclip = 5
        self.test = test

        tmpBasePath = tempfile.mkdtemp()
        if not os.path.isdir(tmpBasePath):
            os.makedirs(tmpBasePath)
        print(tmpBasePath)
        self.out = tmpBasePath
        self.resume = ''

    def run(self, embeddings, dataset=None):
        # TODO: this is ugly hack 
        dataset = "ptb"
        self.unit = embeddings.matrix.shape[1]

        if self.test:
            train = [0, 1, 2, 3, 4, 5]
            val = [0, 1, 2, 3, 4, 5]
            test = [0, 1, 2, 3, 4, 5]
            vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, }
        else:
            # Load the Penn Tree Bank long word sequence dataset
            train, val, test = chainer.datasets.get_ptb_words()
            vocab = chainer.datasets.get_ptb_words_vocabulary()

        id_to_word = {v: k for k, v in vocab.items()}
        n_vocab = max(train) + 1  # train is just an array of integers
        print('#vocab = {}'.format(n_vocab))

        train_iter = ParallelSequentialIterator(train, self.batchsize, self.window_size)
        val_iter = ParallelSequentialIterator(val, 1, self.window_size, repeat=False)
        test_iter = ParallelSequentialIterator(test, 1, self.window_size, repeat=False)

        # merge the vocabulary
        wv = np.zeros((n_vocab, self.unit), dtype=np.float32)
        for i in range(n_vocab):
            # print(id_to_word[i])
            id = embeddings.vocabulary.get_id(id_to_word[i])
            wv[i] = embeddings.matrix[id]

        if self.gpu >= 0:
            chainer.backends.cuda.get_device_from_id(self.gpu).use()
            if wv is not None:
                wv = chainer.backends.cuda.to_gpu(wv)

        # Prepare an RNNLM model
        rnn = RNNForLM(n_vocab, self.unit, self.method, wv, self.window_size)
        model = L.Classifier(rnn)
        model.compute_accuracy = False  # we only want the perplexity
        if self.gpu >= 0:
            # Make a specified GPU current
            chainer.backends.cuda.get_device_from_id(self.gpu).use()
            model.to_gpu()

        # Set up an optimizer
        optimizer = chainer.optimizers.SGD(lr=1.0)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(self.gradclip))

        # Set up a trainer
        if self.method == 'rnn' or self.method == 'lstm':
            updater = BPTTUpdater(train_iter, optimizer, self.bproplen, self.gpu)
        else:
            updater = chainer.training.StandardUpdater(train_iter, optimizer, device=self.gpu)
        trainer = training.Trainer(updater, (self.epoch, 'epoch'), out=self.out)

        eval_model = model.copy()  # Model with shared params and distinct states
        eval_rnn = eval_model.predictor
        # trainer.extend(extensions.Evaluator(
        #     val_iter, eval_model, device=args.gpu,
        #     # Reset the RNN state at the beginning of each evaluation
        #     eval_hook=lambda _: eval_rnn.reset_state()))

        interval = 10 if self.test else 500
        trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                            trigger=(interval, 'iteration')))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration', 'perplexity', 'val_perplexity']
        ), trigger=(interval, 'iteration'))
        trainer.extend(extensions.ProgressBar(
            update_interval=1 if self.test else 10))
        # trainer.extend(extensions.snapshot())
        # trainer.extend(extensions.snapshot_object(
        #     model, 'model_iter_{.updater.iteration}'))
        if self.resume:
            chainer.serializers.load_npz(self.resume, trainer)

        trainer.run()

        # Evaluate the final model
        print('test')
        eval_rnn.reset_state()
        evaluator = extensions.Evaluator(test_iter, eval_model, device=self.gpu)
        eval_result = evaluator()
        print('test perplexity: {}'.format(np.exp(float(eval_result['main/loss']))))

        experiment_setup = self.__dict__
        experiment_setup["embeddings"] = embeddings.metadata
        experiment_setup["category"] = "default"
        experiment_setup["dataset"] = 'ptb'
        experiment_setup["method"] = self.method
        experiment_setup['task'] = 'language_modeling'
        result = {}
        result['experiment_setup'] = experiment_setup
        result['experiment_setup']['default_measurement'] = 'perplexity'
        result['result'] = []
        result['result'] = {"perplexity": np.exp(float(eval_result['main/loss']))}
        return [result]
