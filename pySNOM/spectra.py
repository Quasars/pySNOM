import numpy as np
from enum import Enum
from pySNOM.images import type_from_channelname
from pySNOM.defaults import Defaults

MeasurementModes = Enum(
    "MeasurementModes", ["None", "nanoFTIR", "PsHet", "PTE", "nanoRaman"]
)
DataTypes = Enum("DataTypes", ["Amplitude", "Phase", "Complex", "Topography"])
ChannelTypes = Enum("ChannelTypes", ["None", "Optical", "Mechanical"])
ScanTypes = Enum("ScanTypes", ["Point", "LineScan", "HyperScan"])


class NeaSpectrum:
    """Storing full spectral measurement data and corresponding measurement parameters"""

    def __init__(
        self,
        data: dict,
        parameters: dict,
        scantype="Point",
        filename=None,
        mode="nanoFTIR",
    ):
        self.filename = filename  # Full path with name
        self._parameters = parameters
        self.data = data

        # set measurement mode
        try:
            self._mode = MeasurementModes[mode]
        except ValueError:
            self._mode = MeasurementModes["nanoFTIR"]
            raise ValueError(mode + "is not a measurement mode!")

        try:
            self._scantype = ScanTypes[scantype]
        except ValueError:
            self._scantype = ScanTypes["Point"]
            raise ValueError(scantype + "is not a measurement mode!")

        if parameters:
            try:
                if parameters["PixelArea"][1] == 1 and parameters["PixelArea"][0] == 1:
                    self._scantype = ScanTypes["Point"]
                elif parameters["PixelArea"][1] == 1 or parameters["PixelArea"][0] == 1:
                    self._scantype = ScanTypes["LineScan"]
                else:
                    self._scantype = ScanTypes["HyperScan"]
                self._mode = MeasurementModes[
                    Defaults().spectral_mode_defs[parameters["Scan"]]
                ]
            except:
                raise ValueError("Parameters dictionary is not valid!")
        else:
            # set measurement mode
            try:
                self._mode = MeasurementModes[mode]
            except ValueError:
                self._mode = MeasurementModes["nanoFTIR"]
                raise ValueError(mode + "is not a measurement mode!")
            # set scan type
            try:
                self._scantype = ScanTypes[scantype]
            except ValueError:
                self._scantype = ScanTypes["Point"]
                raise ValueError(scantype + "is not a measurement mode!")

    @property
    def data(self):
        """Property - data (dict with measurement channels)"""
        return self._data

    @data.setter
    def data(self, value):
        """Data setter to reshape properly"""
        self._data = Tools.reshape_spectrum_data(value, self._parameters)

    @property
    def mode(self):
        """Property - mode (MeasurementMode Enum name)"""
        return self._mode.name

    @property
    def scantype(self):
        """Property - scantype (ScanType Enum name)"""
        return self._scantype.name

    @property
    def parameters(self):
        """Property - scantype (ScanType Enum name)"""
        return self._parameters


class SingleChannelSpectrum(NeaSpectrum):
    def __init__(
        self,
        data,
        wndata,
        filename=None,
        parameters=None,
        mode="nanoFTIR",
        channelname="O2A",
    ):
        super().__init__(data, filename, parameters, mode)
        self._wndata = wndata
        self.channel = channelname
        self.order = 0
        self.datatype = None

    @property
    def data(self):
        """Property - data (dict with measurement channels)"""
        return self._data

    @property
    def wndata(self):
        """Property - wavenumber data"""
        return self._wndata

    @property
    def channel(self):
        """Property - channel (string)"""
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value
        self.channeltype, self.order, self.datatype = type_from_channelname(value)


# TRANSFORMATIONS ------------------------------------------------------------------------------------------------------------------
class Transformation:
    def transform(self, data):
        raise NotImplementedError()


class LinearNormalize(Transformation):
    def __init__(self, wavenumber1=0.0, wavenumber2=1000.0, datatype=DataTypes.Phase):
        self.wn1 = wavenumber1
        self.wn2 = wavenumber2
        self.datatype = datatype

    def transform(self, spectrum, wnaxis):
        wn1idx = np.argmin(abs(wnaxis - self.wn1))
        wn2idx = np.argmin(abs(wnaxis - self.wn2))
        m = (spectrum[wn2idx] - spectrum[wn1idx]) / (wnaxis[wn2idx] - wnaxis[wn1idx])
        C = spectrum[wn1idx] - m * wnaxis[wn1idx]

        if self.datatype == DataTypes.Amplitude:
            return spectrum / (m * wnaxis + C)
        else:
            return spectrum - (m * wnaxis + C)


class RotatePhase(Transformation):
    def __init__(self, degree=0.0, wn_ref=1000.0):
        self.wn_ref = wn_ref
        self.degree = degree

    def transform(self, spectrum, wnaxis):
        if not np.iscomplex(spectrum).any():
            spectrum = np.exp(spectrum * complex(1j))

        angles = wnaxis * np.deg2rad(self.degree) / self.wn_ref

        return np.angle(spectrum * np.exp(angles * complex(1j)))


class ShiftPhaseToZero(Transformation):
    """
    Calculates and applies the phase rotation needed to get a flat, levelled phase spectrum.
    Two reference frequencies have to be provided.
    """

    def __init__(self, wavenumber1=1000.0, wavenumber2=2200.0):
        self.wn1 = wavenumber1
        self.wn2 = wavenumber2

    def transform(self, spectrum, wnaxis):
        if not np.iscomplex(spectrum).any():
            spectrum = np.exp(spectrum * complex(1j))

        wn1idx = np.argmin(abs(wnaxis - self.wn1))
        wn2idx = np.argmin(abs(wnaxis - self.wn2))

        theta1 = np.angle(spectrum[wn1idx])
        wn1 = wnaxis[wn1idx]
        spectrum = spectrum * np.exp(-theta1 * complex(1j))

        theta2 = np.angle(spectrum[wn2idx])
        wn2 = wnaxis[wn2idx]

        angles = (wnaxis - wn1) * theta2 / (wn2 - wn1)
        spectrum = np.angle(spectrum * np.exp(-1 * angles * complex(1j)))

        return spectrum


class NormalizeSpectrum(Transformation):
    def __init__(self, datatype=DataTypes.Phase, dounwrap=False):
        self.datatype = datatype
        self.dounwrap = dounwrap

    def transform(self, spectrum, refspectrum):
        if self.datatype == DataTypes.Phase or self.datatype == DataTypes.Topography:
            if self.dounwrap:
                return np.unwrap(spectrum - refspectrum)
            else:
                return spectrum - refspectrum
        else:
            return spectrum / refspectrum


# TOOLS ------------------------------------------------------------------------------------------------------------------
class Tools:
    @staticmethod
    def reshape_spectrum_data(data, params):
        # To compensate for the zero-filling that NeaSpec does
        n = 1
        if params["Scan"] == "Fourier Scan":
            n = 2

        allchannels = list(data.keys())
        if "Depth" in allchannels:
            spectral_depth = len(np.unique(data["Depth"]))
        elif "Index" in allchannels:
            spectral_depth = len(np.unique(data["Index"]))
        elif "Omega" in allchannels:
            spectral_depth = len(np.unique(data["Omega"]))
        else:
            spectral_depth = params["PixelArea"][2] * n

        for channel in allchannels:
            # Point spectrum
            if params["PixelArea"][1] == 1 and params["PixelArea"][0] == 1:
                data[channel] = np.reshape(data[channel], (spectral_depth))

            # Linescan and HyperScan
            else:
                data[channel] = np.reshape(
                    data[channel],
                    (
                        params["PixelArea"][0],
                        params["PixelArea"][1],
                        spectral_depth,
                    ),
                )

        return data
