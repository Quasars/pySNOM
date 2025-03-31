import unittest

import numpy as np

from pySNOM.images import LineLevel, BackgroundPolyFit, SimpleNormalize, DataTypes, AlignImageStack, mask_from_datacondition, dict_from_imagestack


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
        np.testing.assert_almost_equal(out, [[-4., -3., -1.], [0., 1., 3.], [8., 9., 11.]])
        l = LineLevel(method="difference", datatype=DataTypes.Amplitude)
        out = l.transform(d)
        np.testing.assert_almost_equal(
            out, [[0., 0.2, 0.6], [ 2.2222222,  2.7777778,  3.8888889], [8., 9., 11.]]
        )

    def test_masking_mean(self):
        d = np.zeros([8,10])
        d[2:6,3:7] = 1
        mask = np.ones([8,10])
        mask[2:6,3:7] = np.nan

        l = LineLevel(method="mean", datatype=DataTypes.Phase)

        out = l.transform(d)
        np.testing.assert_almost_equal(out[5,0], -0.4)
        out = l.transform(d,mask=mask)
        np.testing.assert_almost_equal(out[5,0], 0.0)

    def test_masking_median(self):
        d = np.zeros([8,10])
        d[2:6,2:9] = 1
        mask = np.ones([8,10])
        mask[2:6,2:9] = np.nan

        l = LineLevel(method="median", datatype=DataTypes.Phase)
        
        out = l.transform(d)
        np.testing.assert_almost_equal(out[5,0], -1.0)
        out = l.transform(d,mask=mask)
        np.testing.assert_almost_equal(out[5,0], 0.0)

    def test_difference_masking(self):
        d = np.zeros([8,10])
        d[2:6,2:9] = 1
        mask = np.ones([8,10])
        mask[2:6,2:9] = np.nan

        l = LineLevel(method="difference", datatype=DataTypes.Phase)
        out = l.transform(d)
        np.testing.assert_almost_equal(out[5,0], 1.0)
        np.testing.assert_almost_equal(out[1,0], -1.0)
        np.testing.assert_almost_equal(out[5,2], 2.0)

class TestBackgroundPolyFit(unittest.TestCase):
    def test_withmask(self):
        d = np.ones([10,10])
        d[4:8,4:8] = 10
        mask = np.ones([10,10])
        mask[4:8,4:8] = np.nan

        t = BackgroundPolyFit(xorder=1,yorder=1,datatype=DataTypes.Phase)
        out = t.transform(d,mask=mask)
        np.testing.assert_almost_equal(out[0,0], 0.0)
        np.testing.assert_almost_equal(out[9,9], 0.0)

        t = BackgroundPolyFit(xorder=1,yorder=1,datatype=DataTypes.Amplitude)
        out = t.transform(d,mask=mask)
        np.testing.assert_almost_equal(out[0,0], 1.0)
        np.testing.assert_almost_equal(out[9,9], 1.0)

    def test_withoutmask(self):
        d = np.ones([10,10])
        d[4:8,4:8] = 10
        mask = np.ones([10,10])
        mask[4:8,4:8] = np.nan

        t = BackgroundPolyFit(xorder=1,yorder=1,datatype=DataTypes.Phase)
        out = t.transform(d)
        np.testing.assert_almost_equal(out[0,0], -0.2975206611570238)
        np.testing.assert_almost_equal(out[9,9], -3.439338842975202)

        t = BackgroundPolyFit(xorder=1,yorder=1,datatype=DataTypes.Amplitude)
        out = t.transform(d)
        np.testing.assert_almost_equal(out[0,0], 0.7707006369426758)
        np.testing.assert_almost_equal(out[9,9], 0.22525876833718098)

class TestHelperFunction(unittest.TestCase):

    def test_mask_from_condition(self):
        d = np.ones([2,2])
        d[0,0] = 0
        out = mask_from_datacondition(d<1)

        np.testing.assert_equal(out[0,0], np.nan)

class TestSimpleNormalize(unittest.TestCase):

    def test_median(self):
        d = np.zeros([8,10])
        d[2:9,1:9] = 1
        mask = np.ones([8,10])
        mask[2:9,1:9] = np.nan

        l = SimpleNormalize(method="median", datatype=DataTypes.Phase)
        
        out = l.transform(d)
        np.testing.assert_almost_equal(out[0,0], -1.0)
        out = l.transform(d,mask=mask)
        np.testing.assert_almost_equal(out[0,0], 0.0)

    def test_mean(self):
        d = np.zeros([8,10])
        d[2:9,1:9] = 1
        mask = np.ones([8,10])
        mask[2:9,1:9] = np.nan

        l = SimpleNormalize(method="mean", datatype=DataTypes.Phase)
        
        out = l.transform(d)
        np.testing.assert_almost_equal(out[0,0], -0.6)
        out = l.transform(d,mask=mask)
        np.testing.assert_almost_equal(out[0,0], 0.0)

    def test_min(self):
        d=np.asarray([1.0, 2.0, 3.0])
        mask = np.asarray([np.nan, 1, 1])

        l = SimpleNormalize(method="min", datatype=DataTypes.Phase)

        out = l.transform(d)
        np.testing.assert_almost_equal(out, [0.0, 1.0, 2.0])
        out = l.transform(d,mask=mask)
        np.testing.assert_almost_equal(out, [-1.0,0.0,1.0])


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

class TestHelperFunctions(unittest.TestCase):
    def test_dictfromimagestack(self):
        stack = [np.zeros((50, 100)), np.zeros((50, 100))]

        out, outparams = dict_from_imagestack(stack,"O2A")

        self.assertEqual(outparams["PixelArea"], [50,100,2])
        self.assertTrue("M" in list(out.keys()))

if __name__ == "__main__":
    unittest.main()
