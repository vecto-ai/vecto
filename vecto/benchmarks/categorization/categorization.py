from ..base import Benchmark
from collections import defaultdict
import csv


class Categorization(Benchmark):
    def __init__(self, normalize=True, ignore_oov=True):
        self.normalize = normalize
        self.ignore_oov = ignore_oov

    def read_test_set(self, path):
        test = defaultdict(lambda: [])
        if path.endswith('.csv'):
            with open(path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                head = True
                for row in reader:
                    if len(row) < 2:
                        continue
                    if not head:
                        category = row[1]
                        word = row[2]
                        test[category].append(word)
                    head = False
        else:
            with open(path) as f:
                for line in f:
                    id, category, word = line.strip().split()
                    test[category].append(word)
        return test
