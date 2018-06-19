#!/usr/bin/env python3
"""Sample script of word embedding model.

This module implements skip-gram model and continuous-bow model.

"""
import argparse
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
from timeit import default_timer as timer
from chainer import reporter
from chainer import training
from chainer.training import extensions
import logging
import os
import vecto
from vecto.vocabulary import Vocabulary
from vecto.vocabulary.vocabulary import create_from_dir, create_ngram_tokens_from_dir, create_from_annotated_dir
from vecto.corpus import load_file_as_ids
from vecto.embeddings.window_iterators import WindowIterator, DirWindowIterator


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dimensions', '-d', default=100, type=int,
                        help='number of dimensions')
    parser.add_argument('--context_type', '-ct', choices=['linear', 'deps'],
                        default='linear',
                        help='context type, for deps context, the annotated corpus is required')
    parser.add_argument('--context_representation', '-cp', choices=['word', 'deps', 'pos', 'posit', 'lr'], # todo lr posit, interation for deps
                        default='word',
                        help='context representation, for deps (dependency information) and ne (named entity), '
                             'the annotated corpus is required')
    parser.add_argument('--window', '-w', default=5, type=int,
                        help='window size')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'],
                        default='skipgram', help='model type ("skipgram", "cbow")')
    parser.add_argument('--subword', '-sw', choices=['none', 'rnn'],
                        default='none',
                        help='specify if subword-level approach should be used ')
    parser.add_argument('--negative-size', default=5, type=int,
                        help='number of negative samples')
    parser.add_argument('--out_type', '-o', choices=['hsm', 'ns', 'original'],
                        default='ns',
                        help='output model type ("hsm": hierarchical softmax, '
                        '"ns": negative sampling, "original": no approximation)')
    parser.add_argument('--path_vocab',
                        default='',
                        help='path to the vocabulary', required=False)
    parser.add_argument('--path_vocab_ngram_tokens',
                        default='',
                        help='path to the vocabulary of ngram tokens (used for subword models)', required=False)
    parser.add_argument('--path_corpus', help='path to the corpus', required=True)
    parser.add_argument('--path_out', help='path to save embeddings', required=True)
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, help='verbose mode', action='store_true', required=False)

    args = parser.parse_args()
    return args


def print_params(args):
    print('GPU: {}'.format(args.gpu))
    print('dimensions: {}'.format(args.dimensions))
    print('Window: {}'.format(args.window))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Training model: {}'.format(args.model))
    print('Output type: {}'.format(args.out_type))
    print('')


class ContinuousBoW(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(ContinuousBoW, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.loss_func = loss_func

    def __call__(self, x, context):
        e = self.embed(context)
        h = F.sum(e, axis=1) * (1. / context.shape[1])
        loss = self.loss_func(h, x)
        reporter.report({'loss': loss}, self)
        return loss


class SkipGram(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(SkipGram, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.loss_func = loss_func

    def getEmbeddings(self):
        return self.embed.W.data

    def getEmbeddings_context(self):
        return self.loss_func.W.data

    def __call__(self, x, context):
        e = self.embed(context)
        shape = e.shape
        x = F.broadcast_to(x[:, None], (shape[0], shape[1]))
        e = F.reshape(e, (shape[0] * shape[1], shape[2]))
        x = F.reshape(x, (shape[0] * shape[1],))
        loss = self.loss_func(e, x)
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


def save_w2v_legacy(args, model, index2word):
    with open('word2vec.model', 'w') as f:
        f.write('%d %d\n' % (len(index2word), args.dimensions))
        w = cuda.to_cpu(model.embed.W.data)
        for i, wi in enumerate(w):
            v = ' '.join(map(str, wi))
            f.write('%s %s\n' % (index2word[i], v))


def create_model(args, net, vocab):
    model =
    model.vocabulary = vocab
    model.metadata["vocabulary"] = vocab.metadata
    model.metadata.update(vars(args))
    model.metadata["vsmlib_version"] = vsmlib.__version__
    model.matrix = cuda.to_cpu(net.getEmbeddings())
    model.matrix_context = cuda.to_cpu(net.getEmbeddings_context())
    return model


def get_data(path, vocab):
    doc = load_file_as_ids(path, vocab)
    doc = doc[doc >= 0]
    # todo smarter split
    train, val = doc[:-1000], doc[-1000:]
    return train, val


def get_loss_func(args, vocab_context):
    word_counts = vocab_context.lst_frequencies
    if args.out_type == 'hsm':
        HSM = L.BinaryHierarchicalSoftmax
        d_counts = {i: word_counts[i] for i in range(len(word_counts))}
        tree = HSM.create_huffman_tree(d_counts)
        loss_func = HSM(args.dimensions, tree)
        loss_func.W.data[...] = 0
    elif args.out_type == 'ns':
        cs = [word_counts[w] for w in range(len(word_counts))]
        loss_func = L.NegativeSampling(args.dimensions, cs, args.negative_size)
        loss_func.W.data[...] = 0
    elif args.out_type == 'original':
        loss_func = SoftmaxCrossEntropyLoss(args.dimensions, vocab_context.cnt_words)

    if loss_func is None:
        raise Exception('Unknown output type: {}'.format(args.out_type))
    return loss_func


def get_model(args, loss_func, vocab, vocab_ngram_tokens):
    model = None
    if args.subword == 'none':
        if args.model == 'skipgram':
            model = SkipGram(vocab.cnt_words, args.dimensions, loss_func)
        if args.model == 'cbow':
            model = ContinuousBoW(vocab.cnt_words, args.dimensions, loss_func)
    if args.subword == "rnn":
        if args.model == 'skipgram':
            model = utils_subword_rnn.SkipGram(vocab, vocab_ngram_tokens, args.dimensions, loss_func)
        if args.model == 'cbow':
            model = utils_subword_rnn.ContinuousBoW(vocab, vocab_ngram_tokens, args.dimensions, loss_func)

    if model is None:
        raise Exception('Unknown model type: {}'.format(args.model))
    return model


def train(args):
    time_start = timer()
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()


    if args.path_vocab == '':
        vocab = create_from_dir(args.path_corpus)
    else:
        vocab = Vocabulary()
        vocab.load(args.path_vocab)
        logger.info("loaded vocabulary")

    if args.context_representation != 'word': # for deps or ner context representation, we need a new context vocab for NS or HSM loss function.
        vocab_context = create_from_annotated_dir(args.path_corpus, representation=args.context_representation)
    else :
        vocab_context = vocab

    vocab_ngram_tokens = None
    if args.subword != 'none':
        vocab_ngram_tokens = Vocabulary()
        vocab_ngram_tokens.load(args.path_vocab_ngram_tokens)


    loss_func = get_loss_func(args, vocab_context)
    model = get_model(args, loss_func, vocab, vocab_ngram_tokens)

    if args.gpu >= 0:
        model.to_gpu()
        logger.debug("model sent to gpu")

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    if os.path.isfile(args.path_corpus):
        train, val = get_data(args.path_corpus, vocab)
        if args.test:
            train = train[:100]
            val = val[:100]
        train_iter = WindowIterator(train, args.window, args.batchsize)
        val_iter = WindowIterator(val, args.window, args.batchsize, repeat=False)
    else:
        train_iter = DirWindowIterator(path=args.path_corpus, vocab=vocab, window_size=args.window, batch_size=args.batchsize)
    updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.path_out)

    if os.path.isfile(args.path_corpus):
        trainer.extend(extensions.Evaluator(val_iter, model, converter=convert, device=args.gpu))
    trainer.extend(extensions.LogReport())
    if os.path.isfile(args.path_corpus):
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
    else:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()
    model = create_model(args, model, vocab)
    time_end = timer()
    model.metadata["execution_time"] = time_end - time_start
    return model


def run(args):
    model = train(args)
    model.metadata["embeddings_type"] = "vanilla"
    model.save_to_dir(args.path_out)
    if model.matrix_context is not None:
        model.matrix = model.matrix_context
        model.metadata["embeddings_type"] = "context"
        model.save_to_dir(os.path.join(args.path_out, 'context'))
    logger.info("model saved to " + args.path_out)


def main():
    args = parse_args()
    print_params(args)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    run(args)


if __name__ == "__main__":
    main()
