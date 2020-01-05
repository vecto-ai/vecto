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


class Sequence_labeling(Benchmark):

    def __init__(self, normalize=True, window_size=2, method='lr'):
        self.normalize = normalize
        self.window_size = window_size
        self.method = method

    def load_data(self, path, task):
        dicts = {}
        dicts['labels2idx'] = {}
        dicts['words2idx'] = {}

        if task == 'pos':
            tag_position = 1
        elif task == 'chunk':
            tag_position = 2
        else:
            tag_position = -1

        for type in ["train", "test", "valid"]:
            with open(os.path.join(path, type + ".txt")) as file_in:
                for line in file_in:
                    if len(line.strip()) is 0:
                        continue
                    lin = line.strip().split()
                    word = lin[0]
                    tag = lin[tag_position]
                    word = word.lower()
                    if word not in dicts['words2idx']:
                        dicts['words2idx'][word] = len(dicts['words2idx'])
                    if tag not in dicts['labels2idx']:
                        dicts['labels2idx'][tag] = len(dicts['labels2idx'])

        out = {}
        for type in ["train", "test", "valid"]:
            w = []
            l = []
            ww = []
            ll = []
            with open(os.path.join(path, type + ".txt")) as f:
                for line in f:
                    if len(line.strip()) is 0:
                        if len(w) > 0:
                            ww.append(w)
                            ll.append(l)
                            w = []
                            l = []
                        continue
                    lin = line.strip().split()
                    word = lin[0].lower()
                    tag = lin[tag_position]
                    w.append(dicts['words2idx'][word])
                    l.append(dicts['labels2idx'][tag])

            out[type] = (ww, ll)
        return out["train"], out["valid"], out["test"], dicts

    def contextwin(self, l, win):
        '''
        win :: int corresponding to the size of the window
        given a list of indexes composing a sentence
        it will return a list of list of indexes corresponding
        to context windows surrounding each word in the sentence
        '''
        assert win >= 1
        l = list(l)
        # print((int)(win/2))
        lpadded = (int)(win) * [0] + l + (int)(win) * [0]
        out = [lpadded[i:i + win * 2 + 1] for i in range(len(l))]
        # print(out)
        assert len(out) == len(l)
        return out

    '''
    Get sequence labeling task's input and output.
    '''

    def getInputOutput(self, lex, y, win, idx2word):
        input = []
        output = []
        for i in range(len(lex)):
            wordListList = self.contextwin(lex[i], win)
            for j in range(len(wordListList)):
                wordList = wordListList[j]
                realWordList = [idx2word[word] for word in wordList]
                input.append(realWordList)
                output.append(y[i][j])
        return input, output

    '''
    get input (X) embeddings
    '''

    def getX(self, input, m):
        x = []
        OOV_count = 0
        token_count = 0
        # print(m.matrix.shape[0])
        random_vector = m.matrix.sum(axis=0) / m.matrix.shape[0]
        # random_vector = m.matrix[0]
        for wordList in input:
            v = []
            for word in wordList:
                if m.has_word(word):
                    wv = m.get_vector(word)
                else:
                    wv = random_vector
                    OOV_count += 1
                token_count += 1
                # if normalize is True:
                #     wv /= np.linalg.norm(wv)
                v.append(wv)
            v = np.array(v).flatten()
            x.append(v)
        print("out of vocabulary rate : %f" % (OOV_count * 1. / token_count))
        print("vocabulary cover rate : %f" % ((token_count - OOV_count) * 1. / token_count))
        return x

    def get_perf(self, filename, options=[]):
        ''' run conlleval.pl perl script to obtain
        precision/recall and F1 score '''
        out = []
        _conlleval = './vecto/benchmarks/sequence_labeling/conlleval.pl'
        proc = subprocess.Popen(["perl", _conlleval] + options, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        with open(filename, 'r') as file:
            stdout, _ = proc.communicate(file.read().encode())
        for line in stdout.decode().split('\n'):
            # print(line)
            if 'accuracy' in line:
                out = line.split()
                # print(out)
                break

        precision = float(out[6][:-2])
        recall = float(out[8][:-2])
        measure = float(out[10])

        return {'p': precision, 'r': recall, 'measure': measure}

    def run_lr(self, embeddings, my_train_x, my_train_y, my_test_x, my_test_y, method, idx2label, dataset, task):
        print(idx2label)
        # fit LR classifier
        if method == 'lr':
            lrc = LogisticRegression(solver="liblinear")
        if method == '2FFNN':
            lrc = MLPClassifier()

        lrc.fit(my_train_x, my_train_y)

        pred_train = lrc.predict(my_train_x)
        pred_test = lrc.predict(my_test_x)
        pred_train = pred_train.tolist()
        pred_test = pred_test.tolist()

        tmpBasePath = tempfile.mkdtemp()
        print(tmpBasePath)

        evalFile = os.path.join(tmpBasePath, 'eval')

        outputFileObject = open(evalFile, 'w')
        index = 0
        for y, p in zip(my_test_y, pred_test):
            y = idx2label[y]
            p = idx2label[p]
            outputFileObject.write("v" + " " + p + " " + y + "\n")
        outputFileObject.close()

        eval_options = []
        if dataset == 'pos':
            eval_options = ['-r']
        res = self.get_perf(evalFile, eval_options)
        # print(res)

        out = {}
        out["result"] = []
        if task == 'pos':
            name = "accuracy"
        else:
            name = "F1_score"

        out['result'] = {}
        out['result'][name] = res['measure']
        out['result']['precision'] = res['p']
        out['result']['recall'] = res['r']
        out['res'] = res
        out['details'] = {}
        out['details']['my_test_y'] = my_test_y
        out['details']['pred_test'] = pred_test
        return out

    def run(self, embs, dataset):
        if self.normalize:
            embs.normalize()
        path_dataset = dataset.path
        # specify the task (can be ner, pos or chunk)
        task = os.path.basename(path_dataset)
        dataset = task

        # get the dataset
        train_set, valid_set, test_set, dic = self.load_data(path_dataset, task)

        idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
        idx2word = dict((k, v) for v, k in dic['words2idx'].items())

        train_lex, train_y = train_set
        valid_lex, valid_y = valid_set
        test_lex, test_y = test_set

        # add validation data to training data.
        train_lex.extend(valid_lex)
        # train_ne.extend(valid_ne)
        train_y.extend(valid_y)

        # vocsize = len(dic['words2idx'])
        # nclasses = len(dic['labels2idx'])
        # print(nclasses)

        # get the training and test's input and output
        my_train_input, my_train_y = self.getInputOutput(train_lex, train_y, self.window_size, idx2word)
        my_train_x = self.getX(my_train_input, embs)
        my_test_input, my_test_y = self.getInputOutput(test_lex, test_y, self.window_size, idx2word)
        my_test_x = self.getX(my_test_input, embs)

        if self.method == 'lr' or self.method == '2FFNN':
            out = self.run_lr(embs, my_train_x, my_train_y, my_test_x, my_test_y, self.method,
                              idx2label, dataset, task)
        if self.method == 'crf':
            # TODO: implement
            pass
        if self.method == 'lstm':
            # TODO: implement
            pass

        experiment_setup = dict()
        experiment_setup["cnt_train_words"] = len(my_train_input)
        experiment_setup["cnt_test_words"] = len(my_test_input)
        experiment_setup["embeddings"] = embs.metadata
        experiment_setup["category"] = "default"
        experiment_setup["dataset"] = task
        experiment_setup["method"] = self.method
        experiment_setup["task"] = "sequence_labeling"
        experiment_setup["timestamp"] = datetime.datetime.now().isoformat()
        out["experiment_setup"] = experiment_setup
        if task == 'pos':
            name = "accuracy"
        else:
            name = "F1_score"
        out['experiment_setup']['default_measurement'] = name

        # TODO: make dict a valid result as well
        return [out]
