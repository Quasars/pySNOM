import unittest

import numpy as np

from pySNOM.images import LineLevel, DataTypes, AlignImageStack


class TestLineLevel(unittest.TestCase):
    def test_median(self):
        d = np.arange(12).reshape(3, -1)[:, [0, 1, 3]]
        l = LineLevel(method="median", datatype=DataTypes.Phase)
        out = l.transform(d)
        np.testing.assert_almost_equal(
            out, [[-1.0, 0.0, 2.0], [-1.0, 0.0, 2.0], [-1.0, 0.0, 2.0]]
        )
        l = LineLevel(method="median", datatype=DataTypes.Amplitude)
        out = l.transform(d)
        np.testing.assert_almost_equal(
            out, [[0.0, 1.0, 3.0], [0.8, 1.0, 1.4], [0.8888889, 1.0, 1.2222222]]
        )

    def test_mean(self):
        d = np.arange(12).reshape(3, -1)[:, [0, 1, 3]]
        l = LineLevel(method="mean", datatype=DataTypes.Phase)
        out = l.transform(d)
        np.testing.assert_almost_equal(
            out,
            [
                [-1.3333333, -0.3333333, 1.6666667],
                [-1.3333333, -0.3333333, 1.6666667],
                [-1.3333333, -0.3333333, 1.6666667],
            ],
        )
        l = LineLevel(method="mean", datatype=DataTypes.Amplitude)
        out = l.transform(d)
        np.testing.assert_almost_equal(
            out,
            [
                [0.0, 0.75, 2.25],
                [0.75, 0.9375, 1.3125],
                [0.8571429, 0.9642857, 1.1785714],
            ],
        )

    def test_difference(self):
        d = np.arange(12).reshape(3, -1)[:, [0, 1, 3]]
        l = LineLevel(method="difference", datatype=DataTypes.Phase)
        out = l.transform(d)
        np.testing.assert_almost_equal(out, [[-4.0, -3.0, -1.0], [0.0, 1.0, 3.0]])
        l = LineLevel(method="difference", datatype=DataTypes.Amplitude)
        out = l.transform(d)
        np.testing.assert_almost_equal(
            out, [[0.0, 0.2, 0.6], [2.2222222, 2.7777778, 3.8888889]]
        )


class TestAlignImageStack(unittest.TestCase):
    def test_stackalignment(self):
        image1 = np.zeros((50, 100))
        image2 = np.zeros((50, 100))
        image1[10:40, 10:40] = 1
        image2[20:50, 20:50] = 1

        aligner = AlignImageStack()
        shifts, crossrect = aligner.calculate([image1, image2])
        np.testing.assert_equal(shifts, [np.asarray([-10.0, -10.0])])
        np.testing.assert_equal(crossrect, [10, 0, 40, 90])

        out = aligner.transform([image1, image2], shifts, crossrect)
        np.testing.assert_equal(np.shape(out), (2, 29, 90))


if __name__ == "__main__":
    unittest.main()
