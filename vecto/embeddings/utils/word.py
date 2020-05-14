#!/usr/bin/env python3
"""Sample script of word embedding model.

This module implements skip-gram model and continuous-bow model.

"""
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
from chainer import reporter
import logging
from vecto.corpus import DirSlidingWindowCorpus
import numpy as np
from vecto.corpus.tokenization import DEFAULT_TOKENIZER, DEFAULT_JAP_TOKENIZER

logger = logging.getLogger(__name__)


class DirWindowIterator(chainer.dataset.Iterator):
    def __init__(self, path, vocab, window_size, batch_size, language='eng', repeat=True):
        self.path = path
        self.vocab = vocab
        self.window_size = window_size - 1
        self.language = language
        if language == 'jap':
            self.dswc = DirSlidingWindowCorpus(self.path, tokenizer=DEFAULT_JAP_TOKENIZER,
                                               left_ctx_size=self.window_size,
                                               right_ctx_size=self.window_size)
        else:
            self.dswc = DirSlidingWindowCorpus(self.path, tokenizer=DEFAULT_TOKENIZER,
                                               left_ctx_size=self.window_size, right_ctx_size=self.window_size)
        self.batch_size = batch_size
        self._repeat = repeat
        self.epoch = 0
        self.is_new_epoch = False
        self.cnt_words_total = 1
        self.cnt_words_read = 0
        logger.debug("created dir window iterator")

    def next_single_sample(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration
        while True:
            try:
                next_value = next(self.dswc)
                self.cnt_words_read += 1
                if self.epoch == 0:
                    self.cnt_words_total += 1
                break
            except StopIteration:
                self.epoch += 1
                self.is_new_epoch = True
                if self.language == 'jap':
                    self.dswc = DirSlidingWindowCorpus(self.path, tokenizer=DEFAULT_JAP_TOKENIZER,
                                                       left_ctx_size=self.window_size,
                                                       right_ctx_size=self.window_size)
                else:
                    self.dswc = DirSlidingWindowCorpus(self.path, tokenizer=DEFAULT_TOKENIZER,
                                                       left_ctx_size=self.window_size, right_ctx_size=self.window_size)
            if self.epoch > 0 and self.cnt_words_total < 3:
                print("corpus empty")
                raise RuntimeError("Corpus is empty")

        self.center = self.vocab.get_id(next_value['current'])
        self.context = [self.vocab.get_id(w) for w in next_value['context']]

        # append -1 to ensure the size of context are equal
        while len(self.context) < self.window_size * 2:
            self.context.append(-1)
        return self.center, self.context

    @property
    def epoch_detail(self):
        return self.cnt_words_read / self.cnt_words_total

    def __next__(self):
        self.is_new_epoch = False
        centers = []
        contexts = []
        for _ in range(self.batch_size):
            center, context = self.next_single_sample()
            centers.append(center)
            contexts.append(context)
        return np.array(centers, dtype=np.int32), np.array(contexts, dtype=np.int32)


class ContinuousBoW(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(ContinuousBoW, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab + 2, n_units, initialW=I.Uniform(1. / n_units)) # plus 2 for OOV and end symbol.
            self.loss_func = loss_func

    def getEmbeddings(self, gpu):
        return self.embed.W.data[2:]  # plus 2 to remove OOV and end symbol.

    def getEmbeddings_context(self):
        return self.loss_func.W.data

    def __call__(self, x, context):
        context = context + 2  # plus 2 for OOV and end symbol.
        e = self.embed(context)
        h = F.sum(e, axis=1) * (1. / context.shape[1])
        loss = self.loss_func(h, x)
        reporter.report({'loss': loss}, self)
        return loss


class SkipGram(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(SkipGram, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab + 2, n_units, initialW=I.Uniform(1. / n_units)) # plus 2 for OOV and end symbol.
            self.loss_func = loss_func

    def getEmbeddings(self, gpu):
        return self.embed.W.data[2:] # plus 2 to remove OOV and end symbol.

    def getEmbeddings_context(self):
        return self.loss_func.W.data

    def __call__(self, center, context):
        context = context + 2  # plus 2 for OOV and end symbol.
        #print("context:", context.shape)
        #print("center:", center.shape)
        emb_context = self.embed(context)
        shape = emb_context.shape
        center = F.broadcast_to(center[:, None], (shape[0], shape[1]))
        emb_context = F.reshape(emb_context, (shape[0] * shape[1], shape[2]))
        center = F.reshape(center, (shape[0] * shape[1],))
        #print(emb_context.shape, center.shape)
        #exit(1)
        loss = self.loss_func(emb_context, center)
        # shouldn't we divide loss by batch size?
        reporter.report({'loss': loss}, self)
        return loss


class SoftmaxCrossEntropyLoss(chainer.Chain):

    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        with self.init_scope():
            self.out = L.Linear(n_in, n_out, initialW=0)

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.out(x), t)


def convert(batch, device):
    center, context = batch
    if device >= 0:
        center = cuda.to_gpu(center)
        context = cuda.to_gpu(context)
    return center, context
