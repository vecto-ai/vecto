from .base import WordEmbeddings
import tables
import math
import warnings
import numpy as np
import brewer2mpl
import os
from vecto.utils.blas import normed
from vecto.vocabulary import Vocabulary
from vecto.utils.data import save_json, load_json, detect_archive_format_and_open


class WordEmbeddingsDense(WordEmbeddings):
    """Stores dense embeddings.

    """

    def cmp_vectors(self, r1, r2):
        c = normed(r1) @ normed(r2)
        if math.isnan(c):
            return 0
        return (c + 1) / 2

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
        # self.name += os.path.basename(os.path.normpath(path))
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
            for i, w in enumerate(self.vocabulary.lst_words):
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
        # self.name += "_normalized"
        self.metadata["normalized"] = True

    def cache_normalized_copy(self):
        if hasattr(self, 'normalized') and self.normalized == True:
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

    def get_most_similar_vectors(self, u, cnt=10):
        scores = np.zeros(self.matrix.shape[0], dtype=np.float32)
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

    def get_vector(self, w):
        i = self.vocabulary.get_id(w)
        if i < 0:
            raise RuntimeError('word do not exist', w)
        row = self.matrix[i]
        return row

    def has_word(self, w):
        i = self.vocabulary.get_id(w)
        if i < 0:
            return False
        return True
