"""Tests for the downloader module."""

import unittest
from vecto.downloader import *


class Tests(unittest.TestCase):
    @classmethod
    def test_downloader(self):
        downloader = Downloader()
        downloader.fetch_metadata(replace=True)

    @classmethod
    def test_dir_structure(self):
        downloader = Downloader()
        downloader.fetch_metadata(replace=True)
        downloader.update_directory_structure()
        dir_structure = downloader.get_resources()
        self.assertEquals(len(list(dir_structure.keys())), 1)
        self.assertEquals(list(dir_structure.keys())[0], 'resources')
        self.assertEquals(len(list(dir_structure.resources.evaluation.intrinsic)), 2)

    @classmethod
    def test_resource_fetching(self):
        downloader = Downloader()
        downloader.fetch_metadata(replace=True)
        downloader.update_directory_structure()
        # dir_structure = downloader.get_resources()
        # downloader.download_resource(dir_structure.resources.evaluation.intrinsic.analogy.en.BATS)
