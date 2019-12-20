import unittest
from io import StringIO
from contextlib import redirect_stdout
from .test_setup import run_module


class Tests(unittest.TestCase):

    def test_cli(self):
        with self.assertRaises(SystemExit):
            sio = StringIO()
            with redirect_stdout(sio):
                run_module('vecto',
                           'WRONG_COMMAND')
