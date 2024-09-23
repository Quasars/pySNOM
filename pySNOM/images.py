import numpy as np
import copy
from enum import Enum
from pySNOM.defaults import Defaults
from gwyfile.objects import GwyDataField

MeasurementModes = Enum('MeasurementModes', ['None','AFM', 'PsHet', 'WLI', 'PTE', 'TappingAFMIR', 'ContactAFM'])
DataTypes = Enum('DataTypes', ['Amplitude', 'Phase', 'Topography'])
ChannelTypes = Enum('ChannelTypes', ['None','Optical','Mechanical'])

# Full measurement data containing all the measurement channels
class Measurement:
    def __init__(self, data, filename=None, info=None, mode="None"):
        self.filename = filename # Full path with name
        self.mode = mode # Measurement mode (PTE, PSHet, AFM, NanoFTIR) - Enum MeasurementModes
        self._data = data # All channels - Dictionary
        self.info = info

    @property
    def mode(self):
        """Property - measurement mode (Enum)"""
        return self._mode
    @mode.setter
    def mode(self, value: str):
        try:
            self._mode = MeasurementModes[value]
        except ValueError:
            self._mode = MeasurementModes["AFM"]
            raise ValueError(value + 'is not a measurement mode!')

    @property
    def data(self):
        """Property - data (dict with GwyDataFields)"""
        return self._data

    @property
    def info(self):
        """Property - info (dictionary)"""
        return self._info
    
    @info.setter
    def info(self, info):
        self._info = info
        if not info == None:
            m = self._info["Scan"]
            self.mode = Defaults().image_mode_defs[m]

    # METHODS --------------------------------------------------------------------------------------
    def extract_channel(self, channelname: str):
        """Returns a single data channel as GwyDataField"""
        channel = self.data[channelname]
        return channel

    def image_from_channel(self, channelname: str):
        """Returns a single Image object with the requred channeldata"""
        channeldata = self.extract_channel(channelname)
        singleimage = GwyImage(channeldata, filename = self.filename, mode = self.mode, channel = channelname, info = self.info)

        return singleimage


# Single image from a single data channel
class Image(Measurement):
    def __init__(self, data, filename=None, mode="AFM", channelname='Z raw', order=0, datatype=DataTypes['Topography'], info=None):
        super().__init__(data, filename, info=info, mode=mode)
        # Describing channel and datatype
        self.channel = channelname # Full channel name
        self.order = int(order)   # Order, nth
        self.datatype = datatype # Amplitude, Phase, Topography - Enum DataTypes
        
        self.data = data
        self.xoff = 0
        self.yoff = 0
        self.xreal = 1
        self.yreal = 1

    @property
    def data(self):
        """Property - data (numpy array)"""
        # Set the data
        return self._data
    
    @data.setter
    def data(self, new_data):
        """Setter for data (optional if you want data to be modifiable later)"""
        self._data = new_data
        self.xres, self.yres = np.shape(new_data)

    @property
    def channel(self):
        """Property - channel (string)"""
        return self._channel
    
    @channel.setter
    def channel(self,value):

        self._channel = value
        self.channeltype, self.order, self.datatype = type_from_channelname(value)

class GwyImage(Image):
    def __init__(self, data, filename=None, mode="AFM", channelname='Z raw', order=0, datatype=DataTypes['Topography'], info=None):
        super().__init__(data, filename=filename, mode=mode, channelname=channelname, order=order, datatype=datatype, info=info)
    
        self.data = data
        self.xoff = data.xoff
        self.yoff = data.yoff
        self.xreal = data.xreal
        self.yreal = data.yreal

    @property
    def data(self):
        """Property - data (numpy array)"""
        # Set the data
        return self._data
    
    @data.setter
    def data(self,value):
        self._data = value.data
        self.xres, self.yres = np.shape(self._data)
        if self._data is None:
            raise ValueError("The provided data object does not contain 'data' attribute")

def type_from_channelname(channelname):
    if channelname[0] == 'O':
        channeltype = ChannelTypes["Optical"]
    elif 'M' in channelname:
        channeltype = ChannelTypes["Mechanical"]
    else:
        channeltype = ChannelTypes["None"]

    if 'Z' in channelname:
        order = 0
    else:
        order = int(channelname[1])
    
    if channelname[2] == 'A':
        datatype = DataTypes["Amplitude"]
    elif 'Z' in channelname:
        datatype = DataTypes["Topography"]
    else:
        datatype = DataTypes["Phase"]

    return channeltype, order, datatype

class Transformation:

    def transform(self, data):
        raise NotImplementedError()


class LineLevel(Transformation):

    def __init__(self, method='median', datatype=DataTypes.Phase):
        self.method = method
        self.datatype = datatype

    def transform(self, data):
        if self.method == 'median':
            norm = np.median(data, axis=1, keepdims=True)
        elif self.method == 'mean':
            norm = np.mean(data, axis=1, keepdims=True)
        elif self.method == 'difference':
            if self.datatype == DataTypes.Amplitude:
                norm = np.median(data[1:] / data[:-1], axis=1, keepdims=True)
            else:
                norm = np.median(data[1:] - data[:-1], axis=1, keepdims=True)
            data = data[:-1]  # difference does not make sense for the last row
        else:
            if self.datatype == DataTypes.Amplitude:
                norm = 1
            else:
                norm = 0

        if self.datatype == DataTypes.Amplitude:
            return data / norm
        else:
            return data - norm

class RotatePhase(Transformation):

    def __init__(self, degree=0.0):
        self.degree = degree

    def transform(self, data):
        # Construct complex dataset
        complexdata = np.exp(data*complex(1j))
        # Rotate and extract phase
        return np.angle(complexdata*np.exp(np.deg2rad(self.degree)*complex(1j)))

class SelfReference(Transformation):

    def __init__(self, referencedata=1, datatype=DataTypes.Phase):
        self.datatype = datatype
        self.referencedata = referencedata
    def transform(self, data):
        if self.datatype == DataTypes.Amplitude:
            return np.divide(data, self.referencedata)
        elif self.datatype == DataTypes.Phase:
            return data-self.referencedata
        else:
            raise RuntimeError("Self-referencing makes only sense for amplitude or phase data")

class SimpleNormalize(Transformation):

    def __init__(self, method='median', value=1.0, datatype=DataTypes.Phase):
        self.method = method
        self.value = value
        self.datatype = datatype

    def transform(self, data):
        match self.method:
            case 'median':
                if self.datatype == DataTypes.Amplitude:
                    return np.divide(data, np.median(data))
                else:
                    return data - np.median(data)
            case 'mean':
                if self.datatype == DataTypes.Amplitude:
                    return np.divide(data, np.mean(data))
                else:
                    return data - np.mean(data)
            case 'manual':
                if self.datatype == DataTypes.Amplitude:
                    return np.divide(data, self.value)
                else:
                    return data - self.value
                
class BackgroundPolyFit(Transformation):

    def __init__(self, xorder=1, yorder=1, datatype=DataTypes.Phase):
        self.xorder = int(xorder)
        self.yorder = int(yorder)
        self.datatype = datatype
        
    def transform(self, data):
        Z = copy.deepcopy(data)
        x = list(range(0, Z.shape[1]))
        y = list(range(0, Z.shape[0]))
        X, Y = np.meshgrid(x, y)
        x, y = X.ravel(), Y.ravel()

        def get_basis(x, y, max_order_x=1, max_order_y=1):
            """Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc."""
            basis = []
            for i in range(max_order_y+1):
                # for j in range(max_order_x - i +1):
                for j in range(max_order_x+1):
                    basis.append(x**j * y**i)
            return basis

        basis = get_basis(x, y, self.xorder, self.yorder)
        A = np.vstack(basis).T
        b = Z.ravel()
        c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

        background = np.sum(c[:, None, None] * np.array(get_basis(X, Y, self.xorder, self.yorder)).reshape(len(basis), *X.shape),axis=0)

        if self.datatype == DataTypes["Amplitude"]:
            return np.divide(Z, background), background
        else:
            return Z-background, background