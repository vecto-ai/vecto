#!/usr/bin/env python3
"""Sample script of word embedding model.

This module implements skip-gram model and continuous-bow model.

"""
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
from chainer import reporter
from vecto.corpus import DirSlidingWindowCorpus
from vecto.vocabulary.vocabulary import get_ngram_tokensList_from_word
import logging
import time
from vecto.corpus.tokenization import DEFAULT_TOKENIZER, DEFAULT_JAP_TOKENIZER

logger = logging.getLogger(__name__)

args = None


class DirWindowIterator(chainer.dataset.Iterator):
    def __init__(self, path, vocab, vocab_ngram_tokens, word2chars, window_size, batch_size, language='eng',
                 repeat=True):
        self.path = path
        self.vocab = vocab
        self.vocab_ngram_tokens = vocab_ngram_tokens
        self.word2chars = word2chars
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
        self.context_left = [0] * window_size
        # self.context_left = collections.deque(maxlen=window_size)
        self.context_right = []
        self.center = 0
        self.cnt_words_total = 1
        self.cnt_words_read = 0
        self.tmp_timestamp = time.time()
        self.tmp_start_timestamp = time.time()
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

        wordIds = self.context
        # print("original context", self.context)
        # words = [self.vocab.get_word_by_id(c) for c in wordIds if c >= 0]
        # TODO: this is ugly hack to avoid <0 check in vocab
        # should use normal ids for <unc> etc
        words = [self.vocab.lst_words[c] for c in wordIds]
        # print("words", words)
        tokenIdsListList = getTokenIdsListList(words, self.vocab_ngram_tokens, self.word2chars)
        # print("#######3tokenIdsListList", self.center, self.context, tokenIdsListList)

        return self.center, self.context, tokenIdsListList

    @property
    def epoch_detail(self):
        return self.cnt_words_read / self.cnt_words_total

    def __next__(self):

        self.is_new_epoch = False
        centers = []
        contexts = []
        tokenIdsListListList = []
        for _ in range(self.batch_size):
            center, context, tokenIdsListList = self.next_single_sample()
            centers.append(center)
            contexts.append(context)
            tokenIdsListListList.append(tokenIdsListList)

        local_max_tokens_length = 7
        for tokenIdsListList in tokenIdsListListList:
            for tokenIdsList in tokenIdsListList:
                for tokenIds in tokenIdsList:
                    if len(tokenIds) > local_max_tokens_length:
                        local_max_tokens_length = len(tokenIds)
        # print(local_max_tokens_length)

        for tokenIdsListList in tokenIdsListListList:
            for tokenIdsList in tokenIdsListList:
                for tokenIds in tokenIdsList:
                    while len(tokenIds) < local_max_tokens_length:
                        tokenIds.append(-2)

        # print(tokenIdsListList)
        tokenIdsListListList = np.array(tokenIdsListListList, dtype=np.int32)
        tokenIdsList_merged = np.reshape(tokenIdsListListList,
                                         (tokenIdsListListList.shape[0] * tokenIdsListListList.shape[1] *
                                          tokenIdsListListList.shape[2],
                                          tokenIdsListListList.shape[3]))
        argsort, argsort_reverse, pList = getChianerInput(tokenIdsList_merged)
        return np.array(centers, dtype=np.int32), np.array(contexts,
                                                           dtype=np.int32), tokenIdsList_merged, argsort, argsort_reverse, pList


def getChianerInput(tokenIdsList_merged):
    non_zero = tokenIdsList_merged > -2
    pList = np.sum(non_zero, axis=0)
    argsort = np.sum(non_zero, axis=1).argsort()[::-1]
    argsort_reverse = argsort.argsort()
    return argsort, argsort_reverse, pList


def convert(batch, device):
    center, context, tokenIdsList_merged, argsort, argsort_reverse, pList = batch
    tokenIdsList_merged_b = get_tokenIdsList_merged_b(tokenIdsList_merged, args.subword)
    if device >= 0:
        center = cuda.to_gpu(center)
        context = cuda.to_gpu(context)
        tokenIdsList_merged = cuda.to_gpu(tokenIdsList_merged)
        if tokenIdsList_merged_b is not None:
            tokenIdsList_merged_b = cuda.to_gpu(tokenIdsList_merged_b)
        argsort = cuda.to_gpu(argsort)
        argsort_reverse = cuda.to_gpu(argsort_reverse)
        pList = cuda.to_gpu(pList)
    return center, context, tokenIdsList_merged, tokenIdsList_merged_b, argsort, argsort_reverse, pList


def get_subwords_from_word2chars(word, word2chars):
    l = []
    for w in word:
        if w not in word2chars:
            l.append(w)
        else:
            for c in word2chars[w]:
                l.append(c)
    return l


def getTokenIdsListList(words, vocab_ngram_tokens, word2chars, max_tokens_length=20):
    tokenIdsListList = []

    min_ = vocab_ngram_tokens.metadata["min_gram"]
    max_ = vocab_ngram_tokens.metadata["max_gram"]

    for word in words:
        tokenIdsList = []
        tokensList = []
        tokensList.extend(get_ngram_tokensList_from_word(word, min_, max_))
        if word2chars is not None:
            tokensList.append(get_subwords_from_word2chars(word, word2chars))
        # print(tokensList)
        for tokens in tokensList:
            tokenIds = [vocab_ngram_tokens.get_id(token) for token in tokens]
            while len(tokenIds) > max_tokens_length:
                del tokenIds[-1]
            tokenIdsList.append(tokenIds)

        tokenIdsListList.append(tokenIdsList)
    return tokenIdsListList


class CNN1D(chainer.Chain):
    def __init__(self, vocab, vocab_ngram_tokens, n_units, n_units_char,
                 dropout, subword):  # dropout ratio, zero indicates no dropout
        super(CNN1D, self).__init__()
        with self.init_scope():
            self.subword = subword
            # n_units_char = 15
            self.embed = L.EmbedID(
                len(vocab_ngram_tokens.lst_words) + 2, n_units_char,
                initialW=I.Uniform(1. / n_units_char))  # ngram tokens embedding  plus 2 for OOV and end symbol.

            self.n_ngram = vocab_ngram_tokens.metadata["max_gram"] - vocab_ngram_tokens.metadata["min_gram"] + 1

            # n_filters = {i: min(200, i * 5) for i in range(1, 1 + 1)}
            # self.cnns = (L.Convolution2D(1, v, (k, n_units_char),) for k, v in n_filters.items())
            # self.out = L.Linear(sum([v for k, v in n_filters.items()]), n_units)
            if 'small' in self.subword:
                self.cnn1 = L.ConvolutionND(1, n_units_char, 50, (1,), )
                self.out = L.Linear(50, n_units)
            else:
                self.cnn1 = L.ConvolutionND(1, n_units_char, 50, (1,), )
                self.cnn2 = L.ConvolutionND(1, n_units_char, 100, (2,), )
                self.cnn3 = L.ConvolutionND(1, n_units_char, 150, (3,), )
                self.cnn4 = L.ConvolutionND(1, n_units_char, 200, (4,), )
                self.cnn5 = L.ConvolutionND(1, n_units_char, 200, (5,), )
                self.cnn6 = L.ConvolutionND(1, n_units_char, 200, (6,), )
                self.cnn7 = L.ConvolutionND(1, n_units_char, 200, (7,), )
                self.out = L.Linear(1100, n_units)

            self.dropout = dropout
            self.vocab = vocab
            self.vocab_ngram_tokens = vocab_ngram_tokens

    def __call__(self, tokenIdsList_merged, tokenIdsList_merged_b, argsort, argsort_reverse,
                 pList):  # input a list of token ids, output a list of word embeddings
        tokenIdsList_merged += 2
        input_emb = self.embed(tokenIdsList_merged)
        # input = input.reshape(input.shape[0], input.shape[1], input.shape[2])
        input_emb = F.transpose(input_emb, (0, 2, 1))
        input_emb = F.dropout(input_emb, self.dropout)
        # print(input.shape)
        if 'small' in self.subword:
            h = self.cnn1(input_emb)
            h = F.max(h, (2,))
        else:
            h1 = self.cnn1(input_emb)
            h1 = F.max(h1, (2,))
            h2 = self.cnn2(input_emb)
            h2 = F.max(h2, (2,))
            h3 = self.cnn3(input_emb)
            h3 = F.max(h3, (2,))
            h4 = self.cnn4(input_emb)
            h4 = F.max(h4, (2,))
            h5 = self.cnn5(input_emb)
            h5 = F.max(h5, (2,))
            h6 = self.cnn6(input_emb)
            h6 = F.max(h6, (2,))
            h7 = self.cnn7(input_emb)
            h7 = F.max(h7, (2,))
            h = F.concat((h1, h2, h3, h4, h5, h6, h7))

        h = F.dropout(h, self.dropout)
        h = F.tanh(h)
        y = self.out(h)
        # print(y.shape)
        e = y
        e = F.reshape(e, (int(e.shape[0] / self.n_ngram),
                          self.n_ngram, e.shape[1]))
        e = F.sum(e, axis=1)
        return e


class RNN(chainer.Chain):
    def __init__(self, vocab, vocab_ngram_tokens, n_units, n_units_char, dropout,
                 subword):  # dropout ratio, zero indicates no dropout
        super(RNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(
                len(vocab_ngram_tokens.lst_words) + 2, n_units_char,
                initialW=I.Uniform(1. / n_units_char))  # ngram tokens embedding  plus 2 for OOV and end symbol.
            if 'lstm' in subword:
                self.mid = L.LSTM(n_units_char, n_units_char * 2)
            self.out = L.Linear(n_units_char * 2, n_units_char)  # the feed-forward output layer
            if 'bilstm' in subword:
                self.mid_b = L.LSTM(n_units_char, n_units_char * 2)
                self.out_b = L.Linear(n_units_char * 2, n_units_char)

            self.n_ngram = vocab_ngram_tokens.metadata["max_gram"] - vocab_ngram_tokens.metadata["min_gram"] + 1
            self.final_out = L.Linear(n_units * (self.n_ngram), n_units)

            self.dropout = dropout
            self.vocab = vocab
            self.vocab_ngram_tokens = vocab_ngram_tokens
            self.subword = subword

    def reset_state(self):
        self.mid.reset_state()
        if 'bilstm' in self.subword:
            self.mid_b.reset_state()

    def __call__(self, tokenIdsList_merged, tokenIdsList_merged_b, argsort, argsort_reverse,
                 pList):  # input a list of token ids, output a list of word embeddings
        tokenIdsList_ordered = tokenIdsList_merged[argsort]
        tokenIdsList_ordered += 2

        if tokenIdsList_merged_b is not None:
            tokenIdsList_ordered_b = tokenIdsList_merged_b[argsort]
            tokenIdsList_ordered_b += 2

        self.reset_state()
        y = None
        for i in range(tokenIdsList_ordered.shape[1]):
            if pList[i] == 0:
                break
            if i == 0:
                self.rnn(tokenIdsList_ordered[:, i])
                if 'sum' in self.subword:
                    y = F.dropout(self.mid.h, self.dropout)
                    # y = self.out(y)
            else:
                self.rnn(tokenIdsList_ordered[0: pList[i], i])
                if 'sum' in self.subword:
                    tmp_y = F.dropout(self.mid.h, self.dropout)
                    # tmp_y = self.out(tmp_y)
                    if pList[i] < tmp_y.shape[0]:
                        tmp_y = tmp_y[0: pList[i], :]
                        tmp_y = (tmp_y + y[0: pList[i], :])
                        y = F.concat((tmp_y, y[pList[i]:, :]), axis=0)
                    else:
                        tmp_y = (tmp_y + y)
                        y = tmp_y

                    # print(tokenIdsList_ordered_b.shape)
                    # print(tokenIdsList_ordered)
                    # print(tokenIdsList_ordered_b)

        if 'bilstm' in self.subword:
            y_b = None
            for i in range(tokenIdsList_ordered_b.shape[1]):
                if pList[i] == 0:
                    break
                if i == 0:
                    self.rnn_b(tokenIdsList_ordered_b[:, i])
                    if 'sum' in self.subword:
                        y_b = F.dropout(self.mid.h, self.dropout)
                else:
                    self.rnn_b(tokenIdsList_ordered_b[0: pList[i], i])
                    if 'sum' in self.subword:
                        tmp_y = F.dropout(self.mid.h, self.dropout)
                        if pList[i] < tmp_y.shape[0]:
                            tmp_y = tmp_y[0: pList[i], :]
                            tmp_y = (tmp_y + y_b[0: pList[i], :])
                            y_b = F.concat((tmp_y, y_b[pList[i]:, :]), axis=0)
                        else:
                            tmp_y = (tmp_y + y_b)
                            y_b = tmp_y

        if 'sum' not in self.subword:  # pure lstm, without sum/avg over all timestep
            y = F.dropout(self.mid.h, self.dropout)
            if 'bilstm' in self.subword:
                y_b = F.dropout(self.mid_b.h, self.dropout)

        y = self.out(y)
        if 'bilstm' in self.subword:
            y_b = self.out_b(y_b)
            y = y + y_b

        e = y[argsort_reverse]

        # isSum = True
        # if isSum:
        #     e = F.reshape(e, (int(e.shape[0] / self.n_ngram),
        #                       self.n_ngram, e.shape[1]))
        #     e = F.sum(e, axis=1)
        # else:
        #     e = F.reshape(e, (int(e.shape[0] / self.n_ngram),
        #                       self.n_ngram * e.shape[1]))
        #     e = self.final_out(F.tanh(e))

        e = F.reshape(e, (int(e.shape[0] / self.n_ngram),
                          self.n_ngram, e.shape[1]))
        e = F.sum(e, axis=1)
        return e

    def rnn(self, cur_word):
        # Given the current word ID, predict the next word.
        x = self.embed(cur_word)
        x = F.dropout(x, self.dropout)
        return self.mid(x)

    def rnn_b(self, cur_word):
        # Given the current word ID, predict the next word.
        x = self.embed(cur_word)
        x = F.dropout(x, self.dropout)
        return self.mid_b(x)


class SUMAVG(chainer.Chain):
    def __init__(self, vocab, vocab_ngram_tokens, n_units, n_units_char, dropout,
                 subword):  # dropout ratio, zero indicates no dropout
        super(SUMAVG, self).__init__()
        with self.init_scope():
            if subword.startswith('sum'):
                self.f_sumavg = F.sum
            if subword.startswith('avg'):
                self.f_sumavg = F.average

            self.embed = L.EmbedID(
                len(vocab_ngram_tokens.lst_words) + 2, n_units_char,
                initialW=I.Uniform(1. / n_units_char))  # ngram tokens embedding  plus 2 for OOV and end symbol.

            self.n_ngram = vocab_ngram_tokens.metadata["max_gram"] - vocab_ngram_tokens.metadata["min_gram"] + 1
            self.dropout = dropout
            self.vocab = vocab
            self.vocab_ngram_tokens = vocab_ngram_tokens

    def __call__(self, tokenIdsList_merged, tokenIdsList_merged_b, argsort, argsort_reverse,
                 pList):  # input a list of token ids, output a list of word embeddings
        tokenIdsList_ordered = tokenIdsList_merged[argsort]
        tokenIdsList_ordered += 2

        start = int(0)
        while pList[start] == tokenIdsList_ordered.shape[0]:
            start = start + 1
        block = tokenIdsList_ordered[pList[start]:, 0:start + 1]
        y = self.f_sumavg(self.embed(block), axis=1)
        for i in range(start, len(pList)):
            if pList[i] == 0:
                break
            if i + 1 < len(pList):
                l = pList[i + 1]
            else:
                l = 0
            r = pList[i]
            if l == r:
                continue
            block = tokenIdsList_ordered[l:r, 0:i + 1]
            t = self.f_sumavg(self.embed(block), axis=1)
            y = F.concat((t, y), axis=0)
        # print(y.shape)
        e = y[argsort_reverse]
        e = F.reshape(e, (int(e.shape[0] / self.n_ngram),
                          self.n_ngram, e.shape[1]))
        e = self.f_sumavg(e, axis=1)
        return e


def get_tokenIdsList_merged_b(tokenIdsList_merged, subword):
    if 'bilstm' in subword:
        tokenIdsList_merged_b = None
        # re-order for backward direction
        for i in range(tokenIdsList_merged.shape[0]):
            t = tokenIdsList_merged.shape[1] - 1
            while tokenIdsList_merged[i][t] == 0 and t >= 1:
                t = t - 1
            # print(t)
            # print(tokenIdsList_ordered.shape)
            t = t
            t = np.concatenate((tokenIdsList_merged[i][:t][::-1], tokenIdsList_merged[i][t:]), axis=0)
            t = t.reshape(1, t.shape[0])
            if tokenIdsList_merged_b is None:
                tokenIdsList_merged_b = t
            else:
                tokenIdsList_merged_b = np.concatenate((tokenIdsList_merged_b, t), axis=0)
        return tokenIdsList_merged_b
    return None


class SkipGram(chainer.Chain):
    def __init__(self, subword, vocab, vocab_ngram_tokens, dimensions, loss_func,
                 dropout=0):  # dropout ratio, zero indicates no dropout
        super(SkipGram, self).__init__()

        with self.init_scope():
            self.subword = subword
            self.vocab = vocab
            self.vocab_ngram_tokens = vocab_ngram_tokens
            self.n_ngram = vocab_ngram_tokens.metadata["max_gram"] - vocab_ngram_tokens.metadata["min_gram"] + 1

            if 'none' in subword:
                self.word_embed = L.EmbedID(len(vocab.lst_words) + 2, dimensions, initialW=I.Uniform(1. / dimensions)) # plus 2 for OOV and end symbol.
            else:
                self.word_embed = None

            if subword.startswith('_none'):
                self.f = None
            # if subword.startswith('cnn_'):
            #     self.f = CNN(vocab, vocab_ngram_tokens, dimensions, dimensions, dropout)
            if subword.startswith('cnn1d'):
                self.f = CNN1D(vocab, vocab_ngram_tokens, dimensions, dimensions, dropout, args.subword)
            if subword.startswith('bilstm') or subword.startswith('lstm'):
                self.f = RNN(vocab, vocab_ngram_tokens, dimensions, dimensions, dropout, args.subword)
            if subword.startswith('avg') or subword.startswith('sum'):
                self.f = SUMAVG(vocab, vocab_ngram_tokens, dimensions, dimensions, dropout, args.subword)

            self.loss_func = loss_func

    def getEmbeddings(self, gpu):
        if self.word_embed is None:
            return self.getEmbeddings_f(gpu=gpu)
        return self.word_embed.W.data[2:] # plus 2 to remove OOV and end symbol.

    def getEmbeddings_f(self, words=None, batchsize=1000, gpu=-1):
        if self.f is None:
            return None
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            if words is None:
                words = self.vocab.lst_words
            i_words = 0
            e = None
            while i_words < len(words):
                tokenIdsListList = getTokenIdsListList(words[i_words: i_words + batchsize],
                                                       self.vocab_ngram_tokens, None)

                local_max_tokens_length = 7
                for tokenIdsList in tokenIdsListList:
                    for tokenIds in tokenIdsList:
                        if len(tokenIds) > local_max_tokens_length:
                            local_max_tokens_length = len(tokenIds)
                for tokenIdsList in tokenIdsListList:
                    for tokenIds in tokenIdsList:
                        while len(tokenIds) < local_max_tokens_length:
                            tokenIds.append(-2)

                tokenIdsListList = np.array(tokenIdsListList, dtype=np.int32)

                tokenIdsList_merged = np.reshape(tokenIdsListList,
                                                 (tokenIdsListList.shape[0] * tokenIdsListList.shape[1],
                                                  tokenIdsListList.shape[2],))
                argsort, argsort_reverse, pList = getChianerInput(tokenIdsList_merged)

                tokenIdsList_merged_b = get_tokenIdsList_merged_b(tokenIdsList_merged, self.subword)
                if gpu >= 0:
                    tokenIdsList_merged = cuda.to_gpu(tokenIdsList_merged)
                    if tokenIdsList_merged_b is not None:
                        tokenIdsList_merged_b = cuda.to_gpu(tokenIdsList_merged_b)
                    argsort = cuda.to_gpu(argsort)
                    argsort_reverse = cuda.to_gpu(argsort_reverse)
                    pList = cuda.to_gpu(pList)
                e_batch = self.f(tokenIdsList_merged, tokenIdsList_merged_b, argsort, argsort_reverse, pList)
                if e is None:
                    e = e_batch
                else:
                    e = F.concat((e, e_batch), axis=0)
                i_words += args.batchsize
        return e.data

    def getEmbeddings_context(self):
        return self.loss_func.W.data

    def __call__(self, x, context, tokenIdsList_merged, tokenIdsList_merged_b, argsort, argsort_reverse, pList):
        # print(tokenIdsListListList.shape) # #batchs * #contexts * #ngrams　* #tokens
        n_batch = x.shape[0]
        n_ngram = self.n_ngram
        n_context = int(len(tokenIdsList_merged) / n_ngram / n_batch)

        x = F.broadcast_to(x[:, None], (n_batch, n_context))
        x = F.reshape(x, (n_batch * n_context,))
        # print(x.shape)

        loss_total = 0
        if self.f is not None:
            e = self.f(tokenIdsList_merged, tokenIdsList_merged_b, argsort, argsort_reverse, pList)
            # print(e.shape)
            loss = self.loss_func(e, x)
            reporter.report({'loss': loss}, self)
            loss_total += loss

        if self.word_embed is not None:
            context = context + 2 # plus 2 for OOV and end symbol.
            e = self.word_embed(context)
            e = F.reshape(e, (e.shape[0] * e.shape[1], e.shape[2]))
            # print(e.shape)
            loss = self.loss_func(e, x)
            reporter.report({'loss': loss}, self)
            loss_total += loss

        return loss_total
