import unittest
import os

import numpy as np

import pySNOM
from pySNOM import readers
from pySNOM.interferograms import (
    NeaInterferogram,
    ProcessSingleChannel,
    ProcessMultiChannels,
    ProcessAllPoints,
)


class test_Neaspectrum(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        fdata = "datasets/testifg_singlepoint.txt"
        data_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], fdata))
        data, measparams = data_reader.read()
        self.ifg = NeaInterferogram(data, measparams, filename=fdata)

        f = "datasets/testifg_multipoints.txt"
        file_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()
        self.multi_ifg = NeaInterferogram(data, params)

    def test_pointinterferogram_object(self):
        np.testing.assert_almost_equal(self.ifg.data["O2A"][100], 9.564073)
        np.testing.assert_string_equal(self.ifg.parameters["Scan"], "Fourier Scan")
        np.testing.assert_string_equal(self.ifg.scantype, "Point")
        np.testing.assert_equal(np.shape(self.ifg.data["O2A"])[0], 2048)

    def test_multipointinterferogram_object(self):
        np.testing.assert_array_equal(
            self.multi_ifg.data["Run"][1, 0], np.asarray([0, 0, 0, 1, 1, 1])
        )

    def test_singlechannel_process(self):
        a2, p2, wn2 = ProcessSingleChannel(order=2, simpleoutput=True).transform(
            self.ifg
        )

        np.testing.assert_almost_equal(a2[500], 61.86039771056218)
        np.testing.assert_almost_equal(p2[500], -2.226779496601322)
        np.testing.assert_almost_equal(wn2[500], 1273.6684340046547)

    def test_multichannel_process(self):
        s_multichannel = ProcessMultiChannels(apod=True).transform(self.ifg)

        np.testing.assert_almost_equal(
            s_multichannel.data["O3A"][500], 23.338979395903074
        )
        np.testing.assert_almost_equal(
            s_multichannel.data["O3P"][500], 0.26155207185723767
        )
        np.testing.assert_almost_equal(
            s_multichannel.data["Wavenumber"][500], 1273.6684340046547
        )

    def test_fullauto_process(self):
        s_allpoints = ProcessAllPoints().transform(self.ifg)

        np.testing.assert_almost_equal(s_allpoints.data["O3A"][500], 23.338979395903074)
        np.testing.assert_almost_equal(
            s_allpoints.data["O3P"][500], 0.26155207185723767
        )
        np.testing.assert_almost_equal(
            s_allpoints.data["Wavenumber"][500], 1273.6684340046547
        )


if __name__ == "__main__":
    unittest.main()
