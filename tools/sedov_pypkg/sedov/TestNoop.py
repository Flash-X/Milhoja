#!/usr/bin/env python3

"""
A do-nothing unittest that is a placeholder used for setting up the automatic
unittest framework.
"""

import unittest

from pathlib import Path

import sedov

FILE_PATH = Path(__file__).parent
TEST_PATH = FILE_PATH.joinpath('TestData')
DATA_PATH = FILE_PATH.joinpath('PkgData')

class TestNoop(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testNoop(self):
        self.assertTrue(True)

def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( TestNoop ) )
    return suite

if __name__ == "__main__":
    unittest.TextTestRunner( verbosity=2 ).run( suite() )

