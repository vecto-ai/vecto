import unittest
import os
from vecto.config import load_config


class Tests(unittest.TestCase):

    @unittest.skipUnless(os.environ.get('CI'), 'skipping as local config likely exists')
    def test_file_corpus(self):
        default_dir = os.path.expanduser("~/.vecto/")
        os.makedirs(default_dir, exist_ok=True)
        path_config = os.path.join(default_dir, 'config.py')
        with self.assertRaises(RuntimeError):
            load_config()
        if not os.path.isfile(path_config):
            with open(path_config, "w") as f:
                f.write("test=1")
        load_config()
