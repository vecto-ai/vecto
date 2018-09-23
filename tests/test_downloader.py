"""Tests for the downloader module."""

import unittest
from vecto.downloader import *


class Tests(unittest.TestCase):

    def test_downloader(self):
        downloader = Downloader()
        downloader.fetch_metadata()

    def test_dir_structure(self):
        downloader = Downloader()
        downloader.fetch_metadata()
        downloader.update_directory_structure()
        dir_structure = downloader.get_resources()
        self.assertEquals(len(list(dir_structure.keys())), 1)
        self.assertEquals(list(dir_structure.keys())[0], 'resources')
        self.assertEquals(len(list(dir_structure.resources.datasets)), 3)

    def test_resource_fetching(self):
        downloader = Downloader()
        downloader.fetch_metadata()
        downloader.update_directory_structure()
        dir_structure = downloader.get_resources()
        downloader.download_resource(dir_structure.resources.datasets.BATS)

