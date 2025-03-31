import numpy as np
import copy
import re
from enum import Enum
from pySNOM.defaults import Defaults

from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, phase_cross_correlation
from scipy.ndimage import fourier_shift

MeasurementModes = Enum(
    "MeasurementModes",
    ["None", "AFM", "PsHet", "WLI", "PTE", "TappingAFMIR", "ContactAFM"],
)
DataTypes = Enum("DataTypes", ["Amplitude", "Phase", "Topography"])
ChannelTypes = Enum("ChannelTypes", ["None", "Optical", "Mechanical"])


# Full measurement data containing all the measurement channels
class Measurement:
    def __init__(self, data, filename=None, info=None, mode="None"):
        self.filename = filename
        self.mode = mode
        self._data = data
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
            raise ValueError(value + "is not a measurement mode!")

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
        singleimage = GwyImage(
            channeldata,
            filename=self.filename,
            mode=self.mode,
            channel=channelname,
            info=self.info,
        )

        return singleimage


# Single image from a single data channel
class Image(Measurement):
    def __init__(
        self,
        data,
        filename=None,
        mode="AFM",
        channelname="Z raw",
        order=0,
        datatype=DataTypes["Topography"],
        info=None,
    ):
        super().__init__(data, filename, info=info, mode=mode)
        # Describing channel and datatype
        self.channel = channelname  # Full channel name
        self.order = int(order)  # Order, nth
        self.datatype = datatype  # Amplitude, Phase, Topography - Enum DataTypes

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
    def channel(self, value):
        self._channel = value
        self.channeltype, self.order, self.datatype = type_from_channelname(value)


class GwyImage(Image):
    def __init__(
        self,
        data,
        filename=None,
        mode="AFM",
        channelname="Z raw",
        order=0,
        datatype=DataTypes["Topography"],
        info=None,
    ):
        super().__init__(
            data,
            filename=filename,
            mode=mode,
            channelname=channelname,
            order=order,
            datatype=datatype,
            info=info,
        )

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
    def data(self, value):
        self._data = value.data
        self.xres, self.yres = np.shape(self._data)
        if self._data is None:
            raise ValueError(
                "The provided data object does not contain 'data' attribute"
            )


def type_from_channelname(channelname):
    channel_strings = ["M(.?)A", "M(.?)P", "O(.?)A", "O(.?)P", "Z C", "Z raw"]
    for pattern in channel_strings:
        if re.search(pattern, channelname) is not None:
            channel_name = re.search(pattern, channelname)[0]

    if channel_name[0] == "O":
        channeltype = ChannelTypes["Optical"]
    elif "M" in channel_name:
        channeltype = ChannelTypes["Mechanical"]
    else:
        channeltype = ChannelTypes["None"]

    if "Z" in channel_name:
        order = 0
    else:
        order = int(channel_name[1])

    if channel_name[2] == "A":
        datatype = DataTypes["Amplitude"]
    elif "Z" in channel_name:
        datatype = DataTypes["Topography"]
    else:
        datatype = DataTypes["Phase"]

    return channeltype, order, datatype


class Transformation:

    def transform(self, data):
        raise NotImplementedError()


class MaskedTransformation(Transformation):

    def calculate(self, data, mask=None):
        raise NotImplementedError()
    
    def correct(self, data, correction):
        """ Applies the calculated corrections to the data """
        if self.datatype == DataTypes.Amplitude:
            return data / correction
        else:
            return data - correction

    def transform(self, data, mask=None):
        """ Calculates and applies the corrections to the data taking into account the mask if given """
        correction = self.calculate(data, mask=mask)
        return self.correct(data, correction)


class LineLevel(MaskedTransformation):
    def __init__(self, method="median", datatype=DataTypes.Phase):
        self.method = method
        self.datatype = datatype

    def calculate(self, data, mask = None):
        if mask is not None:
            data = mask*data

        if self.method == 'median':
            norm = np.nanmedian(data, axis=1, keepdims=True)
        elif self.method == "mean":
            norm = np.nanmean(data, axis=1, keepdims=True)
        elif self.method == "difference":
            if self.datatype == DataTypes.Amplitude:
                norm = np.nanmedian(data[1:] / data[:-1], axis=1, keepdims=True)
                norm = np.append(norm,1)
            else:
                norm = np.nanmedian(data[1:] - data[:-1], axis=1, keepdims=True)
                norm = np.append(norm,0)  # difference does not make sense for the last row
            norm = np.reshape(norm, (norm.size,1))
        else:
            if self.datatype == DataTypes.Amplitude:
                norm = 1
            else:
                norm = 0

        return norm


class RotatePhase(Transformation):
    def __init__(self, degree=0.0):
        self.degree = degree

    def transform(self, data):
        # Construct complex dataset
        complexdata = np.exp(data * complex(1j))
        # Rotate and extract phase
        return np.angle(complexdata * np.exp(np.deg2rad(self.degree) * complex(1j)))


class SelfReference(Transformation):
    def __init__(self, referencedata=1, datatype=DataTypes.Phase):
        self.datatype = datatype
        self.referencedata = referencedata

    def transform(self, data):
        if self.datatype == DataTypes.Amplitude:
            return data / self.referencedata
        elif self.datatype == DataTypes.Phase:
            return data - self.referencedata
        else:
            raise RuntimeError(
                "Self-referencing makes only sense for amplitude or phase data"
            )


class SimpleNormalize(MaskedTransformation):
    def __init__(self, method="median", value=1.0, datatype=DataTypes.Phase):
        self.method = method
        self.value = value
        self.datatype = datatype

    def calculate(self, data, mask=None):
        """ Calculates and returns the image corrections using mask (if given) without applying it to the data"""
        if mask is not None:
            data = mask*data

        match self.method:
            case 'median':
                norm = np.nanmedian(data)
            case 'mean':
                norm = np.nanmean(data)
            case 'manual':
                norm = self.value
            case 'min':
                norm = np.nanmin(data)

        return norm
                

class BackgroundPolyFit(MaskedTransformation):
    def __init__(self, xorder=1, yorder=1, datatype=DataTypes.Phase):
        self.xorder = int(xorder)
        self.yorder = int(yorder)
        self.datatype = datatype

    def calculate(self, data, mask = None):
        """ Calculates and returns the fitted polynomial background using mask (if given) without applying it to the data"""
        if mask is not None:
            data = mask*data

        Z = copy.deepcopy(data)
        x = list(range(0, Z.shape[1]))
        y = list(range(0, Z.shape[0]))
        X, Y = np.meshgrid(x, y)
        x, y = X.ravel(), Y.ravel()
        b = Z.ravel()
        notnanidxs = np.argwhere(~np.isnan(b))
        b = np.ravel(b[notnanidxs])
        x = np.ravel(x[notnanidxs])
        y = np.ravel(y[notnanidxs])

        def get_basis(x, y, max_order_x=1, max_order_y=1):
            """Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc."""
            basis = []
            for i in range(max_order_y + 1):
                # for j in range(max_order_x - i +1):
                for j in range(max_order_x + 1):
                    basis.append(x**j * y**i)
            return basis

        try:
            basis = get_basis(x, y, self.xorder, self.yorder)
            A = np.vstack(basis).T
            c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

            background = np.sum(c[:, None, None] * np.array(get_basis(X, Y, self.xorder, self.yorder)).reshape(len(basis), *X.shape),axis=0)
            
        except ValueError:
            background = np.ones(np.shape(data))
            print("X and Y order must be integer!")

        return background

        
# TODO: Helper functions to create masks or turn other types of masks into 1/Nan mask
def mask_from_booleans(bool_mask, bad_values = False):
    """ Turn a boolean array to an array conatining nans and ones"""
    mshape = np.shape(bool_mask)
    return np.where(bool_mask==bad_values,np.nan*np.ones(mshape),np.ones(mshape))

def mask_from_datacondition(condition):
    mshape = np.shape(condition)
    return np.where(condition,np.nan*np.ones(mshape),np.ones(mshape))


class CalculateOpticalFlow(Transformation):
    """Calculates the pixel coordiate drifts between reference and template image"""

    def __init__(self, image_ref):
        self.image_ref = image_ref

    def transform(self, image):
        v, u = optical_flow_tvl1(
            self.image_ref / np.nanmax(self.image_ref), image / np.nanmax(image)
        )
        return v, u


class WrapImage(Transformation):
    """Applies the pixel-by-pixel drift correction calculated by OpticalFlow"""

    def __init__(self, v, u):
        self.v = v
        self.u = u

    def transform(self, image):
        nr, nc = image.shape
        row_coords, col_coords = np.meshgrid(
            np.arange(nr), np.arange(nc), indexing="ij"
        )
        return warp(
            image, np.array([row_coords + self.v, col_coords + self.u]), mode="edge"
        )


class CalculateXCorrDrift(Transformation):
    """Calculates the drift between reference and template image"""

    def __init__(self, image_ref):
        self.image_ref = image_ref

    def transform(self, image):
        shift, _, _ = phase_cross_correlation(self.image_ref, image)
        return shift


class CorrectImageDrift(Transformation):
    """Rearranges image pixels to correct image shift calculated by cross-correlation"""

    def __init__(self, shift):
        self.shift = shift

    def transform(self, image):
        offset_phase = fourier_shift(np.fft.fftn(image), self.shift)
        offset_phase = np.fft.ifftn(offset_phase)
        return offset_phase.real


class AlignImageStack(Transformation):
    """Calculates the drift between the given images and organize the comman areas into an aligned stack"""

    def __init__(self):
        pass

    def calculate(self, images):
        shifts = []
        crossrect = [0, 0, np.shape(images[0])[0], np.shape(images[0])[1]]
        if len(images) > 1:
            xcorr = CalculateXCorrDrift(images[0])
            for i in range(len(images)):
                if i > 0:
                    shifts.append(xcorr.transform(images[i]))
                    crossrect = shifted_cross_section(
                        rect1=crossrect,
                        rect2=[
                            -shifts[-1][0],
                            shifts[-1][1],
                            np.shape(images[i])[0],
                            np.shape(images[i])[1],
                        ],
                    )
            return shifts, crossrect
        else:
            return None

    def transform(self, images, shifts, crossrect):
        aligned_stack = []
        for i in range(len(images)):
            if i > 0:
                shifter = CorrectImageDrift(shifts[i - 1])
                aligned_stack.append(shifter.transform(images[i]))
                aligned_stack[i] = cut_image(aligned_stack[i], crossrect)
            else:
                aligned_stack.append(cut_image(images[i], crossrect))
        return aligned_stack


def sort_image_stack(images, wns):
    """Sort the image stack based on the wavenumber list"""

    idxs = np.argsort(np.asarray(wns))
    images = [images[i] for i in idxs]
    wns = [wns[i] for i in idxs]

    return images, wns


def create_nparray_stack(measlist):
    """Creates a numpy array stack from a list of measurements, organized as [ rows, columns, wavelengths ] (compatible with quasar io utils)"""

    stack = np.zeros(
        (np.shape(measlist[0])[0], np.shape(measlist[0])[1], len(measlist))
    )

    for i, meas in enumerate(measlist):
        stack[:, :, i] = meas

    return stack


def dict_from_imagestack(X, channelname, wn=None, is_interferogram = True):
    """Converts the image stack into a pySNOM spectra or interferograms compatible dictionary"""
    final_dict = {}
    params = {}

    X = np.asarray(X)

    params["PixelArea"] = [X.shape[1], X.shape[2], X.shape[0]]
    params["Averaging"] = 1
    params["Scan"] = "Fourier Scan"

    final_dict[channelname] = flatten_stack(X)

    y_loc = np.repeat(np.arange(X.shape[1]), X.shape[2])
    x_loc = np.tile(np.arange(X.shape[2]), X.shape[1])

    final_dict["Row"] = np.repeat(y_loc, X.shape[0])
    final_dict["Column"] = np.repeat(x_loc, X.shape[0])

    if is_interferogram:
        depth_channel_name = "M"
    else:
        depth_channel_name = "Wavenumber"

    if wn is not None:
        final_dict[depth_channel_name] = np.tile(wn, X.shape[1] * X.shape[2])
    else:
        final_dict[depth_channel_name] = np.tile(
            np.arange(X.shape[0]), X.shape[1] * X.shape[2]
        )

    return final_dict, params

def flatten_stack(imagestack):
    """ Flatten out values in an image stack to be aneble to add it to spectral dictionaries """
    imagestack = np.asarray(imagestack)
    flattened_values = imagestack.reshape((imagestack.shape[0], imagestack.shape[1] * imagestack.shape[2]))
    return np.ravel(flattened_values, order="F")

def shifted_cross_section(rect1: list, rect2: list):
    """Calculates the cross-section of two rectangle shifted to each other"""
    x1 = rect1[1]
    x2 = rect2[1]
    y1 = rect1[0]
    y2 = rect2[0]
    W1 = rect1[3]
    W2 = rect2[3]
    H1 = rect1[2]
    H2 = rect2[2]

    if y2 > y1:
        Hn = H1 - (y2 - y1)
        yn = y2
    elif (y2 < y1) and (y1 + H1 > y2 + H2):  # Negative shift and higher than H2
        Hn = H2 + (y2 - y1)
        yn = y1
    else:
        Hn = H1
        yn = y1

    if x2 > x1:  # Positive shift
        Wn = W1 - (x2 - x1)
        xn = x2
    elif (x2 < x1) and (x1 + W1 > x2 + W2):  # Negative shift and higher than W2
        Wn = W2 + (x2 - x1)
        xn = x1
    else:
        Wn = W1
        xn = x1

    return int(yn), int(xn), int(Hn), int(Wn)


def cut_image(image, rect):
    """Cuts the part of the image array defined by rectangle"""
    return image[-(rect[2]) : -(rect[0] + 1), rect[1] : rect[1] + rect[3]]
