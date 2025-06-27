import unittest
import os

import numpy as np

import pySNOM
from pySNOM import readers, spectra


class test_Neaspectrum(unittest.TestCase):
    def test_pointspectrum_object(self):
        f = "datasets/testspectrum_singlepoint.txt"
        file_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()

        s = spectra.NeaSpectrum(data, params)

        np.testing.assert_almost_equal(s.data["O2A"][0], 0.1600194)
        np.testing.assert_string_equal(s.parameters["Scan"], "Fourier Scan")
        np.testing.assert_string_equal(s.scantype, "Point")
        np.testing.assert_equal(np.shape(s.data["O2A"])[0], 2048)

    def test_multipointspectrum_object(self):
        f = "datasets/testspectrum_multipoint.txt"
        file_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()

        s = spectra.NeaSpectrum(data, params)

        np.testing.assert_almost_equal(s.data["O2A"][1, 0, 0], 0.1600194)
        np.testing.assert_string_equal(s.parameters["Scan"], "Fourier Scan")
        np.testing.assert_string_equal(s.scantype, "LineScan")
        np.testing.assert_equal(np.shape(s.data["O2A"])[2], 4)

    def test_transfromations(self):
        f = "datasets/testspectrum_singlepoint.txt"
        file_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()
        fref = "datasets/testspectrum_singlepoint_ref.txt"
        file_reader_ref = readers.NeaSpectralReader(
            os.path.join(pySNOM.__path__[0], fref)
        )
        data_ref, params_ref = file_reader_ref.read()

        s = spectra.NeaSpectrum(data, params)
        r = spectra.NeaSpectrum(data_ref, params_ref)

        channel = "O2A"
        normdata = spectra.NormalizeSpectrum(spectra.DataTypes.Amplitude).transform(
            s.data[channel], r.data[channel]
        )
        corrdata = spectra.LinearNormalize(
            wavenumber1=1000, wavenumber2=2200, datatype=spectra.DataTypes.Amplitude
        ).transform(normdata, s.data["Wavenumber"])

        np.testing.assert_almost_equal(normdata[1000], 0.7278023)
        np.testing.assert_almost_equal(corrdata[1000], 0.9795999)


if __name__ == "__main__":
    unittest.main()
