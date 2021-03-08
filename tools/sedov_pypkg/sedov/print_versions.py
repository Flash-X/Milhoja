import sys

import yt
import matplotlib

import numpy as np
import pandas as pd

import sedov

def print_versions():
    """
    Print out the version of this package as well as its installed dependences.
    """
    print('python     version\t{}'.format(sys.version))
    print('numpy      version\t{}'.format(np.__version__))
    print('pandas     version\t{}'.format(pd.__version__))
    print('matplotlib version\t{}'.format(matplotlib.__version__))
    print('yt         version\t{}'.format(yt.__version__))
    print('Sedov      version\t{}'.format(sedov.__version__))

