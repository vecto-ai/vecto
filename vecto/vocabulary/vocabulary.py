import os
import numpy as np
import time
import datetime
from vecto._version import VERSION
from vecto.utils.formathelper import countof_fmt
from vecto.utils.data import save_json, load_json
from vecto.corpus import DirTokenIterator, FileTokenIterator
import logging

logger = logging.getLogger(__name__)


class Vocabulary(object):

    def __init__(self):
        # todo: check if our ternary tree module is available
        self.dic_words_ids = {}
        self.lst_words = []
        self.lst_frequencies = []
        self.metadata = {}

    def tokens_to_ids(self, tokens):
        ids = np.ones(len(tokens), dtype=np.int32) * -1
        for i, t in enumerate(tokens):
            ids[i] = self.get_id(t)
        return ids

    def get_id(self, w):
        try:
            return self.dic_words_ids[w]
        except KeyError:
            return -1

    def get_word_by_id(self, i):
        return(self.lst_words[i])

    def get_frequency(self, i):
        if len(self.lst_frequencies) == 0:
            return 0
        if type(i) == str:
            i = self.get_id(i)
        if i < 0:
            return 0
        return(self.lst_frequencies[i])

    def save_to_dir(self, path):
        os.makedirs(path, exist_ok=True)
        f = open(os.path.join(path, "vocab.tsv"), "w")
        f.write("#word\tfrequency\n")
        for i in range(len(self.lst_words)):
            f.write("{}\t{}\n".format(self.lst_words[i], self.lst_frequencies[i]))
        f.close()
        save_json(self.metadata, os.path.join(path, "metadata.json"))

    def load_list_from_sorted_file(self, filename):
        self.lst_words = []
        f = open(filename, encoding='utf-8', errors='replace')
        lines = f.readlines()
        for line in lines:
            token = line.strip()
            self.lst_words.append(token)
        f.close()

    def create_dic_from_list(self):
        self.dic_words_ids = {}
        for i in range(len(self.lst_words)):
            self.dic_words_ids[self.lst_words[i]] = i

    def load_from_list(self, path):
        self.load_list_from_sorted_file(path)
        self.create_dic_from_list()

    def load_tsv(self, path):
        pos = 0
        f = open(os.path.join(path, "vocab.tsv"))
        self.lst_frequencies = []
        self.dic_words_ids = {}
        self.lst_words = []
        for line in f:
            if line.startswith("#"):
                continue
            word, frequency = line.split("\t")
            self.lst_words.append(word)
            self.lst_frequencies.append(int(frequency))
            self.dic_words_ids[word] = pos
            pos += 1
        f.close()
        self.cnt_words = len(self.lst_words)
        self.lst_frequencies = np.array(self.lst_frequencies)
        self.metadata = load_json(os.path.join(path, "metadata.json"))

    def load(self, path):
        if os.path.isfile(os.path.join(path, "vocab.tsv")):
            self.load_tsv(path)

        files = os.listdir(path)
        for f in files:
            if f.endswith(".vocab"):
                logger.info("found vocab file")
                self.load_from_list(os.path.join(path, f))


class Vocabulary_simple(Vocabulary):

    def load_dic_from_file(self, filename):
        rdic = {}
        f = open(os.path.join(self.dir_root, filename),
                 encoding='utf-8', errors='replace')
        lines = f.readlines()
        for line in lines:
            tokens = line.split("\t")
            rdic[tokens[0]] = np.int64(tokens[-1])
        f.close()
        return rdic

    def load_list_from_file(self, filename, n):
        # postfix = 0
        self.lst_words = [""] * n
        # rdic={}
        # rlst=[]
        f = open(os.path.join(self.dir_root, filename),
                 encoding='utf-8', errors='replace')
        lines = f.readlines()
        for line in lines:
            tokens = line.split("\t")
        #    if tokens[0] in rdic:
            # rdic[tokens[0]+str(postfix)+tokens[1]]=np.int64(tokens[-1])
            # postfix+=1
            # else:
            # rdic[tokens[0]]=np.int64(tokens[-1])
            # rlst.append(tokens[0])
            self.lst_words[np.int64(tokens[-1])] = tokens[0]
        f.close()

    def load(self, path, verbose=False):
        self.dir_root = path
        self.dic_words_ids = self.load_dic_from_file("ids")
        self.load_list_from_file("ids", len(self.dic_words_ids))
        if os.path.isfile(os.path.join(path, "freq_per_id")):
            f = open(os.path.join(self.dir_root, "freq_per_id"))
            self.lst_frequencies = np.fromfile(f, dtype=np.uint64)
            f.close()


class Vocabulary_cooccurrence(Vocabulary_simple):

    def load(self, path, verbose=False):
        t_start = time.time()
        Vocabulary_simple.load(self, path)
        t_end = time.time()
        # assert len(self.lst_words)==len(self.dic_words_ids)
        if verbose:
            cnt_words = len(self.lst_words)
            print("Vocabulary loaded in {0:0.2f} seconds".format(
                t_end - t_start))
            print("{} words ({}) in vocabulary".format(
                cnt_words, countof_fmt(cnt_words)))


def _create_from_iterator(iterator, min_frequency=0):
    t_start = time.time()
    dic_freqs = {}
    for w in iterator:
        if w in dic_freqs:
            dic_freqs[w] += 1
        else:
            dic_freqs[w] = 1
    v = Vocabulary_simple()
    v.lst_frequencies = []
    for i, word in enumerate(sorted(dic_freqs, key=dic_freqs.get, reverse=True)):
        frequency = dic_freqs[word]
        if frequency < min_frequency:
            break
        v.lst_frequencies.append(frequency)
        v.lst_words.append(word)
        v.dic_words_ids[word] = i
    v.cnt_words = len(v.lst_words)
    v.metadata["min_frequency"] = min_frequency
    v.metadata["vecto_version"] = VERSION
    v.metadata["cnt_words"] = v.cnt_words
    t_end = time.time()
    v.metadata["execution_time"] = t_end - t_start
    v.metadata["timestamp"] = datetime.datetime.now().isoformat()
    return v


def create_from_dir(path, min_frequency=0):
    """Collects vocabulary from a corpus by a given directory path.
    """
    if not os.path.isdir(path):
        raise RuntimeError("source directory does not exist")
    iter = DirTokenIterator(path)
    v = _create_from_iterator(iter, min_frequency)
    v.metadata["path_source"] = path
    return v


def create_from_file(path, min_frequency=0):
    """Collects vocabulary from a corpus by a given file path.
    """
    if not os.path.isfile(path):
        raise RuntimeError("source file does not exist")
    iter = FileTokenIterator(path)
    v = _create_from_iterator(iter, min_frequency)
    v.metadata["path_source"] = path
    return v


def parse_annotated_token(token):
    if '/' not in token or '[' not in token or ']' not in token:
        raise RuntimeError("annotated format error, should look like 'word/ne[position/deps]'")
    word = token.split('/')[0]
    pos = token.split('/')[1].split('[')[0]
    dep = token.split('[')[1].split(']')[0]
    posit = dep.split('/')[0]
    dep_tag = dep.split('/')[1]
    return word, pos, posit, dep_tag


def get_words_from_annotated_token(token, representation): # the format should look like word/ne[position/deps]

    word, pos, posit, dep_tag = parse_annotated_token(token)

    if representation == 'word':
        return [word]
    if representation == 'pos':
        return [word+'/'+pos]
    if representation == 'deps':
        w1 = word + '/+' + dep_tag
        w2 = word + '/-' + dep_tag
        return [w1, w2]
    raise RuntimeError("no suitable context_representation find. ")


def get_ngram_tokensList_from_word(word, min_gram, max_gram):
    # word = '<' + word + '>'
    word = '<' + word + '>'
    ngram_tokensList = []
    for gram in range(min_gram, max_gram + 1):
        ngram_tokens = []
        for i in range(0, len(word) - gram + 1):
            nt = word[i:i+gram]
            ngram_tokens.append(nt)
        ngram_tokensList.append(ngram_tokens)
    #　print(word)
    # print(ngram_tokensList)
    return ngram_tokensList


def create_from_annotated_dir(path, min_frequency=0, representation='word'): # todo faster creation of vocab
    """Collects vocabulary from a annotated corpus by a given path.

    """
    t_start = time.time()
    dic_freqs = {}
    if not os.path.isdir(path):
        raise RuntimeError("source directory does not exist")
    for token in DirTokenIterator(path, re_pattern = r"[^\s]+"):
        words = get_words_from_annotated_token(token, representation)
        for w in words:
            print(w)
            if w in dic_freqs:
                dic_freqs[w] += 1
            else:
                dic_freqs[w] = 1
    v = Vocabulary_simple()
    v.lst_frequencies = []
    for i, word in enumerate(sorted(dic_freqs, key=dic_freqs.get, reverse=True)):
        frequency = dic_freqs[word]
        if frequency < min_frequency:
            break
        v.lst_frequencies.append(frequency)
        v.lst_words.append(word)
        v.dic_words_ids[word] = i
    v.cnt_words = len(v.lst_words)
    v.metadata["path_source"] = path
    v.metadata["min_frequency"] = min_frequency
    v.metadata["vecto_version"] = VERSION
    v.metadata["cnt_words"] = v.cnt_words
    t_end = time.time()
    v.metadata["execution_time"] = t_end - t_start
    v.metadata["timestamp"] = datetime.datetime.now().isoformat()
    v.metadata["context_representation"] = representation
    return v


def create_ngram_tokens_from_dir(path, min_gram, max_gram, min_frequency=0):
    """Collects ngram tokens from a corpus by a given path.

    """
    t_start = time.time()
    dic_freqs = {}
    if not os.path.isdir(path):
        raise RuntimeError("source directory does not exist")
    for word in DirTokenIterator(path):
        ngram_tokensList = get_ngram_tokensList_from_word(word, min_gram, max_gram)
        for nts in ngram_tokensList:
            for nt in nts:
                if nt in dic_freqs:
                    dic_freqs[nt] += 1
                else:
                    dic_freqs[nt] = 1
    v = Vocabulary_simple()
    v.lst_frequencies = []
    for i, word in enumerate(sorted(dic_freqs, key=dic_freqs.get, reverse=True)):
        frequency = dic_freqs[word]
        if frequency < min_frequency:
            break
        v.lst_frequencies.append(frequency)
        v.lst_words.append(word)
        v.dic_words_ids[word] = i
    v.cnt_words = len(v.lst_words)
    v.metadata["path_source"] = path
    v.metadata["min_frequency"] = min_frequency
    v.metadata["min_gram"] = min_gram
    v.metadata["max_gram"] = max_gram
    v.metadata["vecto_version"] = VERSION
    v.metadata["cnt_words"] = v.cnt_words
    t_end = time.time()
    v.metadata["execution_time"] = t_end - t_start
    v.metadata["timestamp"] = datetime.datetime.now().isoformat()
    return v

