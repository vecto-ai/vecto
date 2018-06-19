import numpy as np
import chainer
from vecto.corpus.iterators import DirTokenIterator
import logging

logger = logging.getLogger(__name__)


class WindowIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, window, batch_size, repeat=True):
        self.dataset = np.array(dataset, np.int32)
        self.window = window
        self.batch_size = batch_size
        self._repeat = repeat

        self.order = np.random.permutation(
            len(dataset) - window * 2).astype(np.int32)
        self.order += window
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i: i_end]
        w = np.random.randint(self.window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
        pos = position[:, None] + offset[None, :]
        context = self.dataset.take(pos)
        center = self.dataset.take(position)

        if i_end >= len(self.order):
            np.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return center, context

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)


class DirWindowIterator(chainer.dataset.Iterator):
    def __init__(self, path, vocab, window_size, batch_size, repeat=True):
        self.path = path
        self.vocab = vocab
        self.token_iter = DirTokenIterator(path)
        self.window_size = window_size - 1
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
        logger.debug("created dir window iterator")

    def next_single_sample(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration
        while True:
            try:
                next_word = next(self.token_iter)
                id_next_word = self.vocab.get_id(next_word)
                if id_next_word >= 0:
                    self.context_right.append(id_next_word)
                self.cnt_words_read += 1
                if self.epoch == 0:
                    self.cnt_words_total += 1
            except StopIteration:
                self.epoch += 1
                self.is_new_epoch = True
                self.token_iter = DirTokenIterator(self.path)
            if self.epoch > 0 and self.cnt_words_total < 3:
                print("corpus empty")
                raise RuntimeError("Corpus is empty")
            if len(self.context_right) > self.window_size:
                break

        self.context_left.append(self.center)
        self.center = self.context_right[0]
        self.context_right = self.context_right[1:]
        self.context_left = self.context_left[-self.window_size:]
        return self.center, self.context_left + self.context_right

    @property
    def epoch_detail(self):
        return self.cnt_words_read / self.cnt_words_total

    def __next__(self):
        self.is_new_epoch = False
        centers = []
        contexts = []
        for i in range(self.batch_size):
            center, context = self.next_single_sample()
            centers.append(center)
            contexts.append(context)
        return np.array(centers, dtype=np.int32), np.array(contexts, dtype=np.int32)
