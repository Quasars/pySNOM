import unittest
import os

import numpy as np

import pySNOM
from pySNOM import NeaImage


class test_public_functions(unittest.TestCase):
    def test_normalize_simple(self):
        f = 'datasets/image.gwy'
        im = NeaImage()
        im.read_from_gwyfile(os.path.join(pySNOM.__path__[0], f), 'M1A raw')

        im2 = pySNOM.NeaImager.normalize_simple(im)

        np.testing.assert_almost_equal(im2.data[4][8], -0.018852233886)
        np.testing.assert_almost_equal(im2.data[7][5],  2.023254394531)

    def self_reference(self):
        # TODO for testing these we need to have a better test file with actual optical data
        f = 'datasets/image.gwy'
        im = NeaImage()
        im.read_from_gwyfile(os.path.join(pySNOM.__path__[0], f), 'M1A raw')

        pass


if __name__ == '__main__':
    unittest.main()
