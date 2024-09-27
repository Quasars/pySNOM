import unittest
import os

import numpy as np

import pySNOM
from pySNOM.readers import NeaSpectrumReader


class test_Neaspectrum(unittest.TestCase):
    def test_readfile(self):
        r = NeaSpectrumReader(os.path.join(pySNOM.__path__[0], 'datasets/sp.txt'))
        data, params = r.read()

        np.testing.assert_almost_equal(data['O1A'][0], 129.49686)

        
if __name__ == '__main__':
    unittest.main()