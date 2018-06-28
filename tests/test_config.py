import unittest
import os
from vecto.config import load_config


class Tests(unittest.TestCase):

    def test_file_corpus(self):
        default_dir = os.path.expanduser("~/.vecto/")
        os.makedirs(default_dir, exist_ok=True)
        path_config = os.path.join(default_dir, 'config.py')
        if not os.path.isfile(path_config):
            with open(path_config, "w") as f:
                f.write("test=1")
        load_config()
