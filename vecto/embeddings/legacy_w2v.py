import os
import numpy as np
from vecto.vocabulary import Vocabulary
from .dense import WordEmbeddingsDense


class ModelW2V(WordEmbeddingsDense):
    """extends dense embeddings to support loading
    of original binary format from Mikolov's w2v"""

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
        # self.name += "_{}".format(size_row)
        self.matrix = np.zeros((cnt_rows, size_row), dtype=np.float32)
        # logger.debug("cnt rows = {}, size row = {}".format(cnt_rows, size_row))
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
        # self.name += "w2v_" + os.path.basename(os.path.normpath(path))
        filename = [file for file in os.listdir(path) if file.endswith("bin")][0]
        self.load_from_file(os.path.join(path, filename))
#        self.load_from_file(os.path.join(path, "vectors.bin"))
        # self.load_provenance(path)
