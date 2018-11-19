"""Tests for the downloader module."""

import unittest
from vecto.downloader import *


class Tests(unittest.TestCase):
    @classmethod
    def test_downloader(self):
        downloader = Downloader()
        downloader.fetch_metadata(replace=False, verbose=False)

    @classmethod
    def test_downloader_with_replace(self):
        downloader = Downloader()
        downloader.fetch_metadata(replace=True, verbose=True)

    def test_dir_structure(self):
        downloader = Downloader()
        downloader.fetch_metadata(replace=True)
        downloader.update_directory_structure()
        dir_structure = downloader.get_resources()
        self.assertEquals(len(list(dir_structure.keys())), 1)

    def test_dir_structure_benchmarks(self):
        downloader = Downloader()
        downloader.fetch_metadata(replace=True)
        downloader.update_directory_structure()
        dir_structure = downloader.get_resources()
        self.assertEquals(list(dir_structure.keys())[0], 'resources')
        self.assertEquals(len(list(dir_structure.resources.evaluation.intrinsic)), 2)
        self.assertEquals(len(list(dir_structure.resources.evaluation.intrinsic.analogy)), 2)

    @classmethod
    def test_resource_fetching(self):
        downloader = Downloader()
        downloader.fetch_metadata(replace=True)
        downloader.update_directory_structure()
        # dir_structure = downloader.get_resources()
        # downloader.download_resource(dir_structure.resources.evaluation.intrinsic.analogy.en.BATS)

    @classmethod
    def test_fetcher(self):
        url = 'https://p-ams2.pcloud.com/D4ZTKr4DFZbBl110ZZZnTDBI7Z2ZZxTkZkZxKzZzkZf7Z17ZOn0J7ZO5owmesMpa5NYaswTUY8ABRfuNdV/BATS_3.0.zip'
        downloader = Downloader()
        downloader.fetch_file(url, downloader.storage_dir, 'test')

    @classmethod
    def test_advanced(self):
        url = 'https://p-ams2.pcloud.com/D4ZTKr4DFZbBl110ZZZnTDBI7Z2ZZxTkZkZxKzZzkZf7Z17ZOn0J7ZO5owmesMpa5NYaswTUY8ABRfuNdV/BATS_3.0.zip'
        downloader = Downloader()
        downloader.fetch_metadata(replace=True)
        downloader.update_directory_structure()
        downloader.fetch_file(url, downloader.storage_dir, 'test')

