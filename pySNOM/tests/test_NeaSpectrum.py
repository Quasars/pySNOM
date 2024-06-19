import unittest
import os

import numpy as np

import pySNOM
from pySNOM import NeaSpectrum


class test_Neaspectrum(unittest.TestCase):
    def test_readfile(self):
        neasp = NeaSpectrum()
        neasp.readNeaSpectrum(os.path.join(pySNOM.__path__[0], 'datasets/sp.txt'))

        np.testing.assert_almost_equal(neasp.data['O1A'][0], 129.49686)

        
if __name__ == '__main__':
    unittest.main()
