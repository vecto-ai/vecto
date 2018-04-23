"""Tests for misc"""
import unittest
import logging
logger = logging.getLogger(__name__)


class Tests(unittest.TestCase):

    def test_import(self):
        logger.info("testing deprecated")
        import vecto
