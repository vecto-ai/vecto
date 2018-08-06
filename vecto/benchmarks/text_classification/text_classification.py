#!/usr/bin/env python
import argparse
import datetime
import json
import os

import chainer
from chainer import training
from chainer.training import extensions
import numpy

from vecto.benchmarks.text_classification import nets
from vecto.benchmarks.text_classification.nlp_utils import convert_seq
from vecto.benchmarks.text_classification import text_datasets
import datetime
from scipy.stats.stats import spearmanr
import os
import math
from ..base import Benchmark
from io import StringIO
from vecto.utils.data import load_json
from vecto.benchmarks.text_classification import nets
from vecto.benchmarks.text_classification import nlp_utils


def load_model(model_path, wv):
    setup = json.load(open(model_path))

    vocab = json.load(open(setup['vocab_path']))
    n_class = setup['n_class']

    # Setup a model
    if setup['model'] == 'rnn':
        Encoder = nets.RNNEncoder
    elif setup['model'] == 'cnn':
        Encoder = nets.CNNEncoder
    elif setup['model'] == 'bow':
        Encoder = nets.BOWMLPEncoder
    encoder = Encoder(n_layers=setup['layer'], n_vocab=len(vocab),
                      n_units=setup['unit'], dropout=setup['dropout'], wv=wv)
    model = nets.TextClassifier(encoder, n_class)
    chainer.serializers.load_npz(setup['model_path'], model)

    gpu = -1  # todo gpu
    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    return model, vocab, setup


def predict(model, sentence):
    model, vocab, setup = model
    sentence = sentence.strip()
    text = nlp_utils.normalize_text(sentence)
    words = nlp_utils.split_text(text, char_based=setup['char_based'])
    xs = nlp_utils.transform_to_array([words], vocab, with_label=False)
    xs = nlp_utils.convert_seq(xs, device=-1, with_label=False)  # todo use GPU
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        prob = model.predict(xs, softmax=True)[0]
    answer = int(model.xp.argmax(prob))
    score = float(prob[answer])
    return answer, score


def get_vectors(model, sentences):
    model, vocab, setup = model
    vectors = []
    for sentence in sentences:
        sentence = sentence.strip()
        text = nlp_utils.normalize_text(sentence)
        words = nlp_utils.split_text(text, char_based=setup['char_based'])
        xs = nlp_utils.transform_to_array([words], vocab, with_label=False)
        xs = nlp_utils.convert_seq(xs, device=-1, with_label=False)  # todo use GPU
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            vector = model.encoder(xs)
            vectors.append(vector.data[0])
    vectors = numpy.asarray(vectors)
    return vectors


class Text_classification(Benchmark):

    def __init__(self, batchsize=64, epoch=5, gpu=-1, layer=1, dropout=0, model=['cnn', 'rnn', 'bow'][1],
                 char_based=False, shrink=100):
        self.current_datetime = '{}'.format(datetime.datetime.today())
        self.batchsize = batchsize
        self.epoch = epoch
        self.gpu = gpu
        self.layer = layer
        self.dropout = dropout
        self.model = model
        self.char_based = char_based
        self.shrink = shrink

    def get_result(self, embeddings, path_dataset, path_output='/tmp/text_classification/'):
        self.out = path_output
        self.unit = embeddings.matrix.shape[1]

        if not os.path.isdir(path_output):
            os.makedirs(path_output)

        # Load a dataset
        self.path_dataset = path_dataset
        if self.path_dataset == 'dbpedia':
            train, test, vocab = text_datasets.get_dbpedia(
                char_based=self.char_based, vocab=embeddings.vocabulary.dic_words_ids, shrink=self.shrink)
        elif self.path_dataset.startswith('imdb.'):
            train, test, vocab = text_datasets.get_imdb(
                fine_grained=self.path_dataset.endswith('.fine'),
                char_based=self.char_based, vocab=embeddings.vocabulary.dic_words_ids, shrink=self.shrink)
        elif self.path_dataset in ['TREC', 'stsa.binary', 'stsa.fine',
                                   'custrev', 'mpqa', 'rt-polarity', 'subj']:
            train, test, vocab = text_datasets.get_other_text_dataset(
                self.path_dataset, char_based=self.char_based, vocab=embeddings.vocabulary.dic_words_ids,
                shrink=self.shrink)
        else:  # finallly, if file is not downloadable, load from local path
            train, test, vocab = text_datasets.get_dataset_from_path(path_dataset,
                                                                     vocab=embeddings.vocabulary.dic_words_ids,
                                                                     char_based=self.char_based, shrink=self.shrink)

        print('# train data: {}'.format(len(train)))
        print('# test  data: {}'.format(len(test)))
        print('# vocab: {}'.format(len(vocab)))
        n_class = len(set([int(d[1]) for d in train]))
        print('# class: {}'.format(n_class))

        train_iter = chainer.iterators.SerialIterator(train, self.batchsize)
        test_iter = chainer.iterators.SerialIterator(test, self.batchsize,
                                                     repeat=False, shuffle=False)

        # Setup a model
        if self.model == 'rnn':
            Encoder = nets.RNNEncoder
        elif self.model == 'cnn':
            Encoder = nets.CNNEncoder
        elif self.model == 'bow':
            Encoder = nets.BOWMLPEncoder
        encoder = Encoder(n_layers=self.layer, n_vocab=len(vocab),
                          n_units=self.unit, dropout=self.dropout, wv=embeddings.matrix)
        model = nets.TextClassifier(encoder, n_class)
        if self.gpu >= 0:
            # Make a specified GPU current
            chainer.backends.cuda.get_device_from_id(self.gpu).use()
            model.to_gpu()  # Copy the model to the GPU

        # Setup an optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

        # Set up a trainer
        updater = training.StandardUpdater(
            train_iter, optimizer,
            converter=convert_seq, device=self.gpu)
        trainer = training.Trainer(updater, (self.epoch, 'epoch'), out=self.out)

        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(
            test_iter, model,
            converter=convert_seq, device=self.gpu))

        # Take a best snapshot
        record_trigger = training.triggers.MaxValueTrigger(
            'validation/main/accuracy', (1, 'epoch'))
        trainer.extend(extensions.snapshot_object(
            model, 'best_model.npz'),
            trigger=record_trigger)

        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

        # Print a progress bar to stdout
        trainer.extend(extensions.ProgressBar())

        # Save vocabulary and model's setting
        if not os.path.isdir(self.out):
            os.mkdir(self.out)
        vocab_path = os.path.join(self.out, 'vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
        model_path = os.path.join(self.out, 'best_model.npz')
        experiment_setup = self.__dict__
        experiment_setup['vocab_path'] = vocab_path
        experiment_setup['model_path'] = model_path
        experiment_setup['n_class'] = n_class
        experiment_setup['datetime'] = self.current_datetime
        with open(os.path.join(self.out, 'args.json'), 'w') as f:
            json.dump(self.__dict__, f)

        # Run the training
        trainer.run()

        result = {}
        result['experiment_setup'] = experiment_setup
        result['experiment_setup']['default_measurement'] = 'accuracy'
        result['experiment_setup']['dataset'] = os.path.basename(os.path.normpath(path_dataset))
        result['experiment_setup']['method'] = self.model
        result['log'] = load_json(os.path.join(self.out, 'log'))

        result['result'] = {"accuracy": result['log'][-1]['validation/main/accuracy']}
        return [result]
