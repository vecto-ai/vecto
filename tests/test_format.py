"""Tests for format module."""

import unittest
from vecto.utils.formathelper import sizeof_fmt, countof_fmt


class Tests(unittest.TestCase):

    def test_sizeof(self):
        val = 12345667
        print("sizeof:", sizeof_fmt(val))
        val = 10.0 ** 32
        print("sizeof:", sizeof_fmt(val))

    def test_countof(self):
        val = 12345667
        print("countof:", countof_fmt(val))
        val = 10.0 ** 32
        print("countof:", countof_fmt(val))
