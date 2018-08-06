"""Tests for datasets"""
import unittest
from vecto.data import Dataset


class Tests(unittest.TestCase):

    def test_datasets(self):
        Dataset("./")

    def test_dataset(self):
        with self.assertRaises(FileNotFoundError):
            Dataset("./path/does/not/exist/")
