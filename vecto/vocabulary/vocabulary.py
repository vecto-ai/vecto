import os
import numpy as np
import time
import datetime
from vecto._version import VERSION
from vecto.utils.formathelper import countof_fmt
from vecto.utils.metadata import WithMetaData
from vecto.corpus import DirCorpus, FileCorpus, ANNOTATED_TEXT_TOKENIZER
from vecto.corpus.tokenization import Tokenizer
import logging

logger = logging.getLogger(__name__)


class Vocabulary(WithMetaData):

    def __init__(self):
        super(Vocabulary, self).__init__()
        # todo: check if our ternary tree module is available
        self.dic_words_ids = {}
        self.lst_words = []
        self.lst_frequencies = []

    @property
    def cnt_words(self):
        return len(self.lst_words)

    def tokens_to_ids(self, tokens):
        ids = np.zeros(len(tokens), dtype=np.int32)
        for i, t in enumerate(tokens):
            ids[i] = self.get_id(t)
        return ids

    def get_id(self, w):
        try:
            return self.dic_words_ids[w]
        except KeyError:
            return 0

    def get_word_by_id(self, i):
        if i < 0:
            raise RuntimeError("word id can not be negative")
        return (self.lst_words[i])

    def get_frequency(self, i):
        if len(self.lst_frequencies) == 0:
            return 0
        if type(i) == str:
            i = self.get_id(i)
        if i < 0:
            return 0
        return (self.lst_frequencies[i])

    def _populate_from_source_and_wordlist(self, source, wordlist):
        self.metadata["source"] = source.metadata
        self.metadata["class"] = "Vocabulary"
        self.metadata["transform"] = "reduced by wordlist"
        i = 0
        for w in source.lst_words:
            if w in wordlist:
                self.lst_words.append(w)
                self.lst_frequencies.append(source.get_frequency(w))
                self.dic_words_ids[w] = i
                i += 1
        self.metadata["cnt_words"] = i

    def filter_by_wordlist(self, wordlist):
        new_vocab = Vocabulary()
        new_vocab._populate_from_source_and_wordlist(self, wordlist)
        return new_vocab

    def save_to_dir(self, path):
        os.makedirs(path, exist_ok=True)
        f = open(os.path.join(path, "vocab.tsv"), "w", encoding="utf8")
        f.write("#word\tfrequency\n")
        len_frequencies = len(self.lst_frequencies)
        for i in range(len(self.lst_words)):
            if len_frequencies > 0:
                f.write("{}\t{}\n".format(self.lst_words[i], self.lst_frequencies[i]))
            else:
                f.write("{}\t{}\n".format(self.lst_words[i], 0))
        f.close()
        self.save_metadata(path)

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
            if len(line) < 2:
                continue
            word, frequency = line.split("\t")
            self.lst_words.append(word)
            self.lst_frequencies.append(int(frequency))
            self.dic_words_ids[word] = pos
            pos += 1
        f.close()
        self.lst_frequencies = np.array(self.lst_frequencies)
        self.init_metadata(base_path=path)

    def load(self, path):

        if os.path.isfile(os.path.join(path, "vocab.tsv")):
            self.load_tsv(path)

        files = os.listdir(path)
        for f in files:
            if f.endswith(".vocab"):
                logger.info("found vocab file")
                self.load_from_list(os.path.join(path, f))

#    def load_dic_from_file(self, filename):
#        rdic = {}
#        f = open(os.path.join(self.dir_root, filename),
#                 encoding='utf-8', errors='replace')
#        lines = f.readlines()
#        for line in lines:
#            tokens = line.split("\t")
#            rdic[tokens[0]] = np.int64(tokens[-1])
#        f.close()
#        return rdic

#    def load_list_from_file(self, filename, n):
#        # postfix = 0
#        self.lst_words = [""] * n
#        # rdic={}
#        # rlst=[]
#        f = open(os.path.join(self.dir_root, filename),
#                 encoding='utf-8', errors='replace')
#        lines = f.readlines()
#        for line in lines:
#            tokens = line.split("\t")
#        #    if tokens[0] in rdic:
#            # rdic[tokens[0]+str(postfix)+tokens[1]]=np.int64(tokens[-1])
#            # postfix+=1
#            # else:
#            # rdic[tokens[0]]=np.int64(tokens[-1])
#            # rlst.append(tokens[0])
#            self.lst_words[np.int64(tokens[-1])] = tokens[0]
#        f.close()

# legacy vsmlib format,
# def load_legacy(self, path, verbose=False):
#     self.dir_root = path
#     self.dic_words_ids = self.load_dic_from_file("ids")
#     self.load_list_from_file("ids", len(self.dic_words_ids))
#     if os.path.isfile(os.path.join(path, "freq_per_id")):
#         f = open(os.path.join(self.dir_root, "freq_per_id"))
#         self.lst_frequencies = np.fromfile(f, dtype=np.uint64)
#         f.close()


def _create_from_iterator(iterator, min_frequency=0):
    t_start = time.time()
    dic_freqs = {}
    for w in iterator:
        if w in dic_freqs:
            dic_freqs[w] += 1
        else:
            dic_freqs[w] = 1
    v = Vocabulary()
    v.lst_frequencies = [0]
    v.lst_words.append("[UNK]")
    v.dic_words_ids["[UNK]"] = 0
    for i, word in enumerate(sorted(dic_freqs, key=dic_freqs.get, reverse=True)):
        frequency = dic_freqs[word]
        if frequency < min_frequency:
            break
        v.lst_frequencies.append(frequency)
        v.lst_words.append(word)
        v.dic_words_ids[word] = i + 1
    v.metadata["min_frequency"] = min_frequency
    v.metadata["cnt_words"] = v.cnt_words
    t_end = time.time()
    v.metadata["execution_time"] = t_end - t_start
    v.metadata["timestamp"] = datetime.datetime.now().isoformat()
    iter_metadata = getattr(iterator, 'metadata', None)
    if iter_metadata is not None:
        v.metadata['source'] = iter_metadata
    return v


def create_from_path(path, min_frequency=0, language='eng'):
    """Collects vocabulary from a corpus by a given directory path.
    """
    tokenizer = Tokenizer(stopwords=[])
    if os.path.isfile(path):
        iter = FileCorpus(path, language).get_token_iterator(tokenizer=tokenizer)
    else:
        if os.path.isdir(path):
            iter = DirCorpus(path, language).get_token_iterator(tokenizer)
        else:
            raise RuntimeError("source path can not be read")
    # TODO: add option for stopwords
    v = _create_from_iterator(iter, min_frequency)
    return v


# TODO: mark as obsolete and remove later
def create_from_file(path, min_frequency=0, language='eng'):
    return create_from_path(path, min_frequency, language)


def create_from_dir(path, min_frequency=0, language='eng'):
    return create_from_path(path, min_frequency, language)


def parse_annotated_token(token):
    if '/' not in token or '[' not in token or ']' not in token:
        raise RuntimeError("annotated format error, should look like 'word/ne[position/deps]'")
    word = token.split('/')[0]
    pos = token.split('/')[1].split('[')[0]
    dep = token.split('[')[1].split(']')[0]
    posit = dep.split('/')[0]
    dep_tag = dep.split('/')[1]
    return word, pos, posit, dep_tag


def get_words_from_annotated_token(token, representation):  # the format should look like word/ne[position/deps]

    word, pos, posit, dep_tag = parse_annotated_token(token)

    if representation == 'word':
        return [word]
    if representation == 'pos':
        return [word + '/' + pos]
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
            nt = word[i:i + gram]
            ngram_tokens.append(nt)
        ngram_tokensList.append(ngram_tokens)
    return ngram_tokensList


def create_from_annotated_dir(path, min_frequency=0, representation='word'):  # todo faster creation of vocab
    """Collects vocabulary from a annotated corpus by a given path.

    """
    t_start = time.time()
    dic_freqs = {}
    if not os.path.isdir(path):
        raise RuntimeError("source directory does not exist")
    # source_corpus = DirTokenCorpus(path, tokenizer=ANNOTATED_TEXT_TOKENIZER)
    source_corpus = DirCorpus(path).get_token_iterator(tokenizer=ANNOTATED_TEXT_TOKENIZER)
    for token in source_corpus:
        words = get_words_from_annotated_token(token, representation)
        for w in words:
            # print(w)
            if w in dic_freqs:
                dic_freqs[w] += 1
            else:
                dic_freqs[w] = 1
    # TODO: does it really differs from _create_from_iterator? maybe merge?
    v = Vocabulary()
    v.lst_frequencies = []
    for i, word in enumerate(sorted(dic_freqs, key=dic_freqs.get, reverse=True)):
        frequency = dic_freqs[word]
        if frequency < min_frequency:
            break
        v.lst_frequencies.append(frequency)
        v.lst_words.append(word)
        v.dic_words_ids[word] = i
    v.metadata["min_frequency"] = min_frequency
    v.metadata["cnt_words"] = v.cnt_words
    t_end = time.time()
    v.metadata["execution_time"] = t_end - t_start
    v.metadata["timestamp"] = datetime.datetime.now().isoformat()
    v.metadata["context_representation"] = representation
    v.metadata["source"] = source_corpus.metadata
    return v


def create_ngram_tokens_from_dir(path, min_gram, max_gram, min_frequency=0):
    """Collects ngram tokens from a corpus by a given path.

    """
    t_start = time.time()
    dic_freqs = {}
    if not os.path.isdir(path):
        raise RuntimeError("source directory does not exist")
    corpus = DirCorpus(path).get_token_iterator()
    for word in corpus:
        ngram_tokensList = get_ngram_tokensList_from_word(word, min_gram, max_gram)
        for nts in ngram_tokensList:
            for nt in nts:
                if nt in dic_freqs:
                    dic_freqs[nt] += 1
                else:
                    dic_freqs[nt] = 1
    v = Vocabulary()
    v.lst_frequencies = []
    for i, word in enumerate(sorted(dic_freqs, key=dic_freqs.get, reverse=True)):
        frequency = dic_freqs[word]
        if frequency < min_frequency:
            break
        v.lst_frequencies.append(frequency)
        v.lst_words.append(word)
        v.dic_words_ids[word] = i
    v.metadata["min_frequency"] = min_frequency
    v.metadata["min_gram"] = min_gram
    v.metadata["max_gram"] = max_gram
    v.metadata["cnt_words"] = v.cnt_words
    t_end = time.time()
    v.metadata["execution_time"] = t_end - t_start
    v.metadata["timestamp"] = datetime.datetime.now().isoformat()
    v.metadata["source"] = corpus.metadata
    return v
