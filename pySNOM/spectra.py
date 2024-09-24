import numpy as np
import os
import copy
from enum import Enum
from scipy import signal
from scipy.fft import fft, fftshift
from scipy.interpolate import CubicSpline, interp1d
import copy

MeasurementModes = Enum('MeasurementModes', ['None','nanoFTIR', 'PsHet', 'PTE'])
DataTypes = Enum('DataTypes', ['Amplitude', 'Phase', 'Topography'])
ChannelTypes = Enum('ChannelTypes', ['None','Optical','Mechanical'])
ScanTypes = Enum('ScanTypes',['Point','LineScan','HyperScan'])

class NeaSpectrum:
    def __init__(self) -> None:
        self.filename = None # Full path with name
        # data from all the channels
        self.data = {}
        # Other parameters from info txt -  Dictionary
        self.parameters = {}

    def saveSpectraToDAT(self,channelname):
        fname = f'{self.filename[0:-4]}.dat'
        M = np.array([self.data["Wavenumber"],self.data[channelname]])
        np.savetxt(fname, M.T)

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
        m = (spectrum[wn2idx] - spectrum[wn1idx])/(wnaxis[wn2idx] - wnaxis[wn1idx])
        C = spectrum[wn1idx] - m*wnaxis[wn1idx]

        if self.datatype == DataTypes.Amplitude:
            return spectrum / (m*wnaxis + C)
        else:
            return spectrum - (m*wnaxis + C)

class RotatePhase(Transformation):

    def __init__(self, value=0.0):
        self.value = value

    def transform(self, spectrum, wnaxis):
        # Construct complex dataset
        complexdata = np.exp(spectrum*complex(1j))
        wnaxis = wnaxis / wnaxis[1]
        # Rotate and extract phase
        angles = wnaxis*self.value
        return np.angle(complexdata*np.exp(angles*complex(1j)))
    
class SpectrumNormalize(Transformation):

    def __init__(self, datatype=DataTypes.Phase, dounwrap=False):
        self.datatype = datatype
        self.dounwrap = dounwrap

    def transform(self, spectrum, refspectrum):
        if self.datatype == DataTypes.Amplitude:
            return np.divide(spectrum,refspectrum)
        else:
            if self.dounwrap:
                return np.unwrap(np.subtract(spectrum,refspectrum))
            else:
                return np.subtract(spectrum,refspectrum)


# TOOLS ------------------------------------------------------------------------------------------------------------------
class Tools:
    def __init__(self):
        pass

    def reshape_linescan(data, parameters):
        return np.reshape(np.ravel(data),(parameters["PixelArea"][0],parameters["PixelArea"][2]))