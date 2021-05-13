"""
Code for working with Sedov test results
"""

import sys
import unittest
import nose.core
import pkg_resources

# constants
from sedov.constants import *

# functions
from sedov.print_versions import print_versions
from sedov.compare_integral_quantities import compare_integral_quantities

# classes
from sedov.Result import Result
from sedov.PacketTimingsSingleFile import PacketTimingsSingleFile
from sedov.PacketTimings import PacketTimings

# visualizations
from sedov.MplConservedQuantities import MplConservedQuantities
from sedov.MplFinalSolution import MplFinalSolution
from sedov.MplSolutionComparison import MplSolutionComparison
from sedov.MplWalltimesByStep import MplWalltimesByStep
from sedov.MplPacketWalltimes import MplPacketWalltimes
from sedov.MplPacketWalltimesByBlock import MplPacketWalltimesByBlock

# unittests
from sedov.TestNoop import suite as suite1

__version__ = pkg_resources.get_distribution('sedov').version

def test_suite():
    return unittest.TestSuite([suite1()])

def test(verbosity=1):
    if verbosity not in [0, 1, 2]:
        msg = 'Invalid verbosity level - {} not in {0,1,2}'
        raise ValueError(msg.format(verbosity))

    print_versions()
    sys.stdout.flush()
    nose.core.TextTestRunner(verbosity=verbosity).run(test_suite())

