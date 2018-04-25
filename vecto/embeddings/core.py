"""The model module that implements embedding loading.
"""

import os
import math
import warnings
import logging
import tables
import brewer2mpl
import numpy as np
import scipy
import scipy.sparse.linalg
from scipy import sparse
from matplotlib import pyplot as plt
from vecto.utils.data import save_json, load_json, detect_archive_format_and_open
from vecto.utils.blas import normed, normalize_sparse
from vecto.vocabulary import Vocabulary_cooccurrence, Vocabulary_simple, Vocabulary
from vecto.utils.sparse import load_matrix_csr

logger = logging.getLogger(__name__)


class Model(object):
    """Basic model class to define interface.

    Usually you would not use this class directly,
    but rather some of the classes which inherit from Model
    """

    def __init__(self):
        self.name = ""
        self.metadata = {}

    # def get_x_label(self, i):
        # return self.vocabulary.get_word_by_id(i)

    def get_most_informative_columns(self, rows, width):
        xdim = rows.shape[1]
        scores = np.zeros(xdim)
        for i in range(rows.shape[0]):
            row = rows[i] / np.linalg.norm(rows[i])
            for j in range(len(row)):
                scores[j] += row[j]
        scores = abs(scores)
        tops = np.argsort(scores)
        return list(reversed(tops[-width:]))

    def filter_rows(self, ids_of_interest):
        # return (cooccurrence[1].todense()[:width])
        xdim = self.matrix.shape[1]
        dense = np.empty([0, xdim])
        # dense=np.empty([0,width])
        for i in ids_of_interest:
            if i < 0:
                continue
            if sparse.issparse(self.matrix):
                row = self.matrix[i].todense()
            else:
                row = self.matrix[i]
            row = np.asarray(row)
            row = np.reshape(row, (xdim))
            # dense=np.vstack([dense,row[:width]])
            dense = np.vstack([dense, row])
        return dense

    def filter_submatrix(self, lst_words_initial, width):
        words_of_interest = [
            w for w in lst_words_initial if self.vocabulary.get_id(w) >= 0]
        ids_of_interest = [self.vocabulary.get_id(
            w) for w in words_of_interest]
        rows = self.filter_rows(ids_of_interest)
        # xdim = rows.shape[1]
        # max_width = 25
        # width=min(xdim,max_width)
        vert = None  # np.empty((rows.shape[0],0))
        cols = self.get_most_informative_columns(rows, width)
        for i in cols:
            if vert is None:
                vert = (rows[:, i])
            else:
                vert = np.vstack([vert, rows[:, i]])
        labels = [self.get_x_label(i) for i in cols]
        return rows, vert.T, labels

    def get_most_similar_vectors(self, u, cnt=10):
        # todo split into dense and sparse implementations
        scores = np.zeros(self.matrix.shape[0], dtype=np.float32)
        if self.normalized:
            scores = normed(u) @ self.matrix.T
            scores = (scores + 1) / 2
        else:
            if hasattr(self, "_normalized_matrix"):
                scores = normed(u) @ self._normalized_matrix.T
                scores = (scores + 1) / 2
            else:
                str_warn = "\n\tthis method executes slow if embeddings are not normalized."
                str_warn += "\n\tuse normalize() method to normalize your embeddings"
                str_warn += "\n\tif for whatever reasons you need your embeddings to be not normalized, you can use .cache_normalized_copy() method to cache normalized copy of embeddings"
                str_warn += "\n\tplease note that latter will consume additional memory\n"
                warnings.warn(str_warn, RuntimeWarning)
                for i in range(self.matrix.shape[0]):
                    scores[i] = self.cmp_vectors(u, self.matrix[i])
        ids = np.argsort(scores)[::-1]
        ids = ids[:cnt]
        return zip(ids, scores[ids])

    def get_most_similar_words(self, w, cnt=10):
        """returns list of words sorted by cosine proximity to a target word

        Args:
            w: target word
            cnt: how many similar words are needed

        Returns:
            list of words and corresponding similarities
        """

        if isinstance(w, str):
            vec = self.matrix[self.vocabulary.get_id(w)]
        else:
            vec = w
        rows = self.get_most_similar_vectors(vec, cnt)
        results = []
        for i in rows:
            results.append([self.vocabulary.get_word_by_id(i[0]), i[1]])
        return results

    def has_word(self, w):
        if self.vocabulary.get_id(w) < 0:
            return False
        return True

    def get_row(self, w):
        i = self.vocabulary.get_id(w)
        if i < 0:
            raise Exception('word do not exist', w)
            # return None
        row = self.matrix[i]
        return row

    def cmp_rows(self, id1, id2):
        r1 = self.matrix[id1]
        r2 = self.matrix[id2]
        return self.cmp_vectors(r1, r2)

    def cmp_words(self, w1, w2):
        id1 = self.vocabulary.get_id(w1)
        id2 = self.vocabulary.get_id(w2)
        if (id1 < 0) or (id2 < 0):
            return 0
        return self.cmp_rows(id1, id2)

    def load_metadata(self, path):
        try:
            self.metadata = load_json(os.path.join(path, "metadata.json"))
        except FileNotFoundError:
            logger.warning("metadata not found")
        if "dimensions" not in self.metadata:
            self.metadata["dimensions"] = self.matrix.shape[1]
        if "vocabulary" in self.metadata:
            self.vocabulary.metadata = self.metadata["vocabulary"]

    @property
    def normalized(self):
        if "normalized" in self.metadata:
            return self.metadata["normalized"]
        return False


class ModelSparse(Model):
    """sparse (usually count-based) embeddings"""
    def __init__(self):
        self.name += "explicit_"

    def cmp_vectors(self, r1, r2):
        distance = r1.dot(r2.T) / (np.linalg.norm(r1.data) * np.linalg.norm(r2.data))
        distance = distance[0, 0]
        if math.isnan(distance):
            return 0
        return (distance + 1) / 2

    def load_from_hdf5(self, path):
        """load model in compressed sparse row format from hdf5 file

        hdf5 file should contain row_ptr, col_ind and data array

        Args:
            path: path to the embeddings folder
        """
        self.load_metadata(path)
        f = tables.open_file(os.path.join(path, 'cooccurrence_csr.h5p'), 'r')
        row_ptr = np.nan_to_num(f.root.row_ptr.read())
        col_ind = np.nan_to_num(f.root.col_ind.read())
        data = np.nan_to_num(f.root.data.read())
        dim = row_ptr.shape[0] - 1
        self.matrix = scipy.sparse.csr_matrix(
            (data, col_ind, row_ptr), shape=(dim, dim), dtype=np.float32)
        f.close()
        self.vocabulary = Vocabulary_cooccurrence()
        self.vocabulary.load(path)
        self.name += os.path.basename(os.path.normpath(path))

    def load(self, path):
        # self.load_provenance(path)
        self.vocabulary = Vocabulary_cooccurrence()
        self.vocabulary.load(path)
        self.name += os.path.basename(os.path.normpath(path))
        self.matrix = load_matrix_csr(path, verbose=True)

    def clip_negatives(self):
        self.matrix.data.clip(0, out=self.matrix.data)
        self.matrix.eliminate_zeros()
        self.name += "_pos"
        self.provenance += "\ntransform : clip negative"

    def normalize(self):
        normalize_sparse(self.matrix)
        self.name += "_normalized"
        self.metadata["normalized"] = True


class ModelDense(Model):
    """Stores dense embeddings.

    """

    def cmp_vectors(self, r1, r2):
        c = normed(r1) @ normed(r2)
        if math.isnan(c):
            return 0
        return (c + 1) / 2

    def save_matr_to_hdf5(self, path):
        f = tables.open_file(os.path.join(path, 'vectors.h5p'), 'w')
        atom = tables.Atom.from_dtype(self.matrix.dtype)
        ds = f.create_carray(f.root, 'vectors', atom, self.matrix.shape)
        ds[:] = self.matrix
        ds.flush()
        f.close()

    def load_hdf5(self, path):
        """loads embeddings from hdf5 format"""
        f = tables.open_file(os.path.join(path, 'vectors.h5p'), 'r')
        self.matrix = f.root.vectors.read()
        self.vocabulary = Vocabulary()
        self.vocabulary.load(path)
        self.name += os.path.basename(os.path.normpath(path))
        f.close()

    def load_npy(self, path):
        """loads embeddings from numpy format"""
        self.matrix = np.load(os.path.join(path, "vectors.npy"))
        # self.load_with_alpha(0.6)
        self.vocabulary = Vocabulary_simple()
        self.vocabulary.load(path)
        self.name += os.path.basename(os.path.normpath(path))

    def save_to_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.vocabulary.save_to_dir(path)
        # self.matrix.tofile(os.path.join(path,"vectors.bin"))
        # np.save(os.path.join(path, "vectors.npy"), self.matrix)
        self.save_matr_to_hdf5(path)
        save_json(self.metadata, os.path.join(path, "metadata.json"))

    def save_to_dir_plain_txt(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'vectors.txt'), 'w') as output:
            for i,w in enumerate(self.vocabulary.lst_words):
                if len(w.strip()) == 0:
                    continue
                output.write(w + ' ')
                for j in range(self.matrix[i].shape[0]):
                    output.write(str(self.matrix[i][j]))
                    output.write(' ')
                output.write("\n")

    def load_with_alpha(self, path, power=0.6):
        # self.load_provenance(path)
        f = tables.open_file(os.path.join(path, 'vectors.h5p'), 'r')
#        left = np.nan_to_num(f.root.vectors.read())
        left = f.root.vectors.read()
        sigma = f.root.sigma.read()
        logger.info("loaded left singular vectors and sigma")
        sigma = np.power(sigma, power)
        self.matrix = np.dot(left, np.diag(sigma))
        logger.info("computed the product")
        self.metadata["pow_sigma"] = power
        self.metadata["size_dimensions"] = int(self.matrix.shape[1])
        f.close()
        self.vocabulary = Vocabulary_simple()
        self.vocabulary.load(path)
        self.name += os.path.basename(os.path.normpath(path)) + "_a" + str(power)

    def normalize(self):
        nrm = np.linalg.norm(self.matrix, axis=1)
        nrm[nrm == 0] = 1
        self.matrix /= nrm[:, np.newaxis]
        self.name += "_normalized"
        self.metadata["normalized"] = True

    def cache_normalized_copy(self):
        if self.normalized:
            self._normalized_matrix = self.matrix
        else:
            self._normalized_matrix = self.matrix.copy()
            self._normalized_matrix /= np.linalg.norm(self._normalized_matrix, axis=1)[:, None]

    def load_from_text(self, path):
        i = 0
        # self.name+="_"+os.path.basename(os.path.normpath(path))
        self.vocabulary = Vocabulary()
        rows = []
        header = False
        vec_size = -1
        with detect_archive_format_and_open(path) as f:
            for line in f:
                tokens = line.split()
                if i == 0 and len(tokens) == 2:
                    header = True
                    cnt_words = int(tokens[0])
                    size_embedding = int(tokens[1])
                    continue
                # word = tokens[0].decode('ascii',errors="ignore")
                # word = tokens[0].decode('UTF-8', errors="ignore")
                word = tokens[0]
                self.vocabulary.dic_words_ids[word] = i
                self.vocabulary.lst_words.append(word)
                str_vec = tokens[1:]
                if vec_size == -1:
                    vec_size = len(str_vec)
                if vec_size != len(str_vec):
                    print(line)
                    continue
                row = np.zeros(len(str_vec), dtype=np.float32)
                for j in range(len(str_vec)):
                    row[j] = float(str_vec[j])
                rows.append(row)
                i += 1
        # if header:
        #     assert cnt_words == len(rows)
        self.matrix = np.vstack(rows)
        if header:
            assert size_embedding == self.matrix.shape[1]
        self.vocabulary.lst_frequencies = np.zeros(len(self.vocabulary.lst_words), dtype=np.int32)
        self.name = os.path.basename(os.path.dirname(os.path.normpath(path)))

    def filter_by_vocab(self, words):
        """reduced embeddings to the provided list of words (which can be empty)

        Args:
            words: set or list of words to keep

        Returns:
            Instance of Dense class

        """
        if len(words) == 0:
            return self
        else:
            new_embds = ModelDense()
            new_embds.vocabulary = Vocabulary()
            lst_new_vectors = []
            i = 0
            for w in self.vocabulary.lst_words:
                if w in words:
                    lst_new_vectors.append(self.get_row(w))
                    new_embds.vocabulary.lst_words.append(w)
                    new_embds.vocabulary.lst_frequencies.append(self.vocabulary.get_frequency(w))
                    new_embds.vocabulary.dic_words_ids[w] = i
                    i += 1
            new_embds.matrix = np.array(lst_new_vectors, dtype=np.float32)
            new_embds.vocabulary.metadata = {}
            new_embds.vocabulary.metadata["cnt_words"] = i
            new_embds.vocabulary.metadata["transform"] = "reduced by wordlist"
            new_embds.vocabulary.metadata["original"] = self.vocabulary.metadata
            new_embds.metadata = self.metadata
            new_embds.metadata["vocabulary"] = new_embds.vocabulary.metadata
            return new_embds


class ModelNumbered(ModelDense):
    """extends dense model by numbering dimensions"""

    def get_x_label(self, i):
        return i

    def viz_wordlist(self, wl, colored=False, show_legend=False):
        colors = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors
        cnt = 0
        for i in wl:
            row = self.get_row(i)
            row = row / np.linalg.norm(row)
            if colored:
                plt.bar(range(0, len(row)), row, color=colors[cnt], linewidth=0, alpha=0.6, label=i)
            else:
                plt.bar(range(0, len(row)), row, color="black", linewidth=0, alpha=1 / len(wl), label=i)
            cnt += 1
        if show_legend:
            plt.legend()


class Model_svd_scipy(ModelNumbered):
    def __init__(self, original, cnt_singular_vectors, power):
        ut, s_ev, _vt = scipy.sparse.linalg.svds(
            original.matrix, k=cnt_singular_vectors, which='LM')  # LM SM LA SA BE
        self.sigma = s_ev
        sigma_p = np.power(s_ev, power)
        self.matrix = np.dot(ut, np.diag(sigma_p))
        self.vocabulary = original.vocabulary
        self.provenance = original.provenance + \
            "\napplied scipy.linal.svd, {} singular vectors, sigma in the power of {}".format(
                cnt_singular_vectors, power)
        self.name = original.name + \
            "_svd_{}_C{}".format(cnt_singular_vectors, power)


class ModelW2V(ModelNumbered):
    """extends ModelDense to support loading of original binary format from Mikolov's w2v"""

    @staticmethod
    def _load_word(file):
        result = b''
        w = b''
        while w != b' ':
            w = file.read(1)
            result = result + w
        return result[:-1]

    def load_from_file(self, filename):
        self.vocabulary = Vocabulary()
        f = open(filename, "rb")
        header = f.readline().split()
        cnt_rows = int(header[0])
        size_row = int(header[1])
        self.name += "_{}".format(size_row)
        self.matrix = np.zeros((cnt_rows, size_row), dtype=np.float32)
        logger.debug("cnt rows = {}, size row = {}".format(cnt_rows, size_row))
        for i in range(cnt_rows):
            word = ModelW2V._load_word(f).decode(
                'UTF-8', errors="ignore").strip()
            self.vocabulary.dic_words_ids[word] = i
            self.vocabulary.lst_words.append(word)
            s_row = f.read(size_row * 4)
            row = np.fromstring(s_row, dtype=np.float32)
            # row = row / np.linalg.norm(row)
            self.matrix[i] = row
        f.close()

    def load_from_dir(self, path):
        self.name += "w2v_" + os.path.basename(os.path.normpath(path))
        filename = [file for file in os.listdir(path) if file.endswith("bin")][0]
        self.load_from_file(os.path.join(path, filename))
#        self.load_from_file(os.path.join(path, "vectors.bin"))
        # self.load_provenance(path)


#@deprecated
#class Model_glove(ModelNumbered):
    #def __init__(self):
        #self.name = "glove"
#
    #def load_from_dir(self, path):
        #self.name = "glove_" + os.path.basename(os.path.normpath(path))
        #files = os.listdir(path)
        #for f in files:
            #if f.endswith(".gz"):
                #logger.info("this is Glove")
                #self.load_from_text(os.path.join(path, f))


def load_from_dir(path):
    """Automatically detects embeddings format and loads

    Args:
        path: directory where embeddings are stores

    Returns:
        Instance of appropriate Model-based class
    """
    if os.path.isfile(os.path.join(path, "cooccurrence_csr.h5p")):
        logger.info("detected as sparse explicit in hdf5")
        result = ModelSparse()
        result.load_from_hdf5(path)
        result.load_metadata(path)
        return result
    if os.path.isfile(os.path.join(path, "bigrams.data.bin")):
        logger.info("detected as sparse in vecto legacy format")
        result = ModelSparse()
        result.load(path)
        result.load_metadata(path)
        return result
    if os.path.isfile(os.path.join(path, "vectors.bin")):
        logger.info("this is w2v original binary format")
        result = ModelW2V()
        result.load_from_dir(path)
        result.load_metadata(path)
        return result
    if os.path.isfile(os.path.join(path, "sgns.words.npy")):
        result = ModelLevy()
        logger.info("this is Levi")
        result.load_from_dir(path)
        result.load_metadata(path)
        return result
    if os.path.isfile(os.path.join(path, "vectors.npy")):
        result = ModelNumbered()
        logger.info("detected as dense ")
        result.load_npy(path)
        result.load_metadata(path)
        return result
    if os.path.isfile(os.path.join(path, "vectors.h5p")):
        result = ModelNumbered()
        logger.info("detected as vecto format ")
        result.load_hdf5(path)
        result.load_metadata(path)
        return result

    result = ModelNumbered()
    files = os.listdir(path)
    for f in files:
        if f.endswith(".gz") or f.endswith(".bz") or f.endswith(".txt"):
            logger.info(path + "Detected VSM in plain text format")
            result.load_from_text(os.path.join(path, f))
            result.load_metadata(path)
            return result
        if f.endswith(".npy"):
            logger.info("Detected VSM in numpy format")
            result.matrix = np.load(os.path.join(path, f))
            result.vocabulary = Vocabulary()
            result.vocabulary.load(path)
            result.load_metadata(path)
            return result
        if any(file.endswith('bin') for file in os.listdir(path)):
            result = ModelW2V()
            logger.info("Detected VSM in the w2v original binary format")
            result.load_from_dir(path)
            result.load_metadata(path)
            return result
#        if f.startswith("words") and f.endswith(".npy") \
#               and os.path.isfile(os.path.join(path, f.replace(".npy", ".vocab"))):
#            result = Model_Fun()
#            result = ModelLevy()
#            logger.info("Detected VSM in npy and vocab in plain text file format")
#            result.load_from_dir(path, f[: -4])
#            result.load_metadata(path)
#            return result

    raise RuntimeError("Cannot detect the format of this VSM")
