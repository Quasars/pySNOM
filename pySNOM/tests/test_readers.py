import unittest
import os

import numpy as np

import pySNOM
from pySNOM import readers


class TestReaders(unittest.TestCase):
    def test_gwyread(self):
        f = "datasets/testPsHetData.gwy"
        file_reader = readers.GwyReader(
            fullfilepath=os.path.join(pySNOM.__path__[0], f), channelname="O3A raw"
        )
        gwy_image = file_reader.read()

        np.testing.assert_almost_equal(gwy_image.data[7][7], 15.213262557983398)
        np.testing.assert_almost_equal(gwy_image.data[8][8], 15.736936569213867)
        np.testing.assert_almost_equal(gwy_image.data[9][9], 13.609171867370605)

    def test_readgsf(self):
        f = "datasets/testPsHet O3A raw.gsf"
        file_reader = readers.GsfReader(os.path.join(pySNOM.__path__[0], f))
        gwy_image = file_reader.read()

        np.testing.assert_almost_equal(gwy_image.data[7][7], 15.213262557983398)
        np.testing.assert_almost_equal(gwy_image.data[8][8], 15.736936569213867)
        np.testing.assert_almost_equal(gwy_image.data[9][9], 13.609171867370605)

    def test_inforead(self):
        f = "datasets/testinfofile.txt"
        file_reader = readers.NeaInfoReader(os.path.join(pySNOM.__path__[0], f))
        data = file_reader.read()

        np.testing.assert_approx_equal(data["ScanArea"][0], 5.000)
        np.testing.assert_almost_equal(data["TargetWavelength"], 6.006)
        np.testing.assert_string_equal(data["DemodulationMode"], "PsHet")

    def test_general_reader_ifg(self):
        f = "datasets/testifg_singlepoint.txt"
        file_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()

        np.testing.assert_almost_equal(data["O2A"][0], 9.580825)
        np.testing.assert_almost_equal(params["Regulator"][0], 3.767854)
        np.testing.assert_string_equal(params["Scan"], "Fourier Scan")

    def test_general_reader_spectrum(self):
        f = "datasets/testspectrum_singlepoint.txt"
        file_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()

        np.testing.assert_almost_equal(data["O2A"][0], 0.1600194)
        np.testing.assert_string_equal(params["Scan"], "Fourier Scan")

    def test_general_reader_nanoraman(self):
        f = "datasets/testspectrum_nanoraman.txt"
        file_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()

        np.testing.assert_almost_equal(data["Data"][0], 11.0)
        np.testing.assert_string_equal(
            params["Scan"], "AFM-Raman/PL Scan (Tapping Mode)"
        )

    def test_legacy_nea_reader(self):
        f = "datasets/neafile_test_ifg.nea"
        file_reader = readers.NeaFileLegacyReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()

        np.testing.assert_almost_equal(data["O2A"][0], 0.8251932)
        np.testing.assert_array_equal(list(data.keys())[-1], "M")
        np.testing.assert_string_equal(params["Scan"], "Fourier Scan")


if __name__ == "__main__":
    unittest.main()
