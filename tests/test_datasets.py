"""Tests for datasets"""
import unittest
from vecto.data import Dataset


class Tests(unittest.TestCase):

    def test_datasets(self):
        Dataset("./")
