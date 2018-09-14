"""Tests for the downloader module."""

import unittest
from vecto.downloader import *


class Tests(unittest.TestCase):

    def test_init(self):
        d = Downloader()
