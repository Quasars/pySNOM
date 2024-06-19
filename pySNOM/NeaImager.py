import gwyfile
import numpy as np
import copy


class NeaImage:
    def __init__(self) -> None:
        # Name regarding file and neaSCAN project
        self.filename = None  # Full path with name
        self.channel_name = None
        self.meas_name = None
        # Important measurement parameters from gwyddion file
        self.xreal = None   # Physical image width
        self.yreal = None   # Physical image height
        self.xoff = None    # Center position X
        self.yoff = None    # Center position Y
        self.xres = None    # Pixel size in X
        self.yres = None    # Pixel size in Y
        self.isamp = None   # Amplitude -True or phase/topography - false - VERY IMPORTANT
        self.istopo = None
        self.isphase = None
        # Data/Image itself
        self.data = None
        # Other parameters from info txt -  Dictionary
        self.parameters = None

    def read_from_gwyfile(self, filename: str, channelname: str):
        self.filename = filename
        self.channel_name = channelname
        gwyobj = gwyfile.load(self.filename)
        channels = gwyfile.util.get_datafields(gwyobj)
        channel = channels[self.channel_name]
        self.isAmplitude()

        # Set the basic attributes from gwyddion field
        for key in channel:
            if key in dir(self):
                setattr(self, key, channel[key])
        self.data = channel.data

    def isAmplitude(self):
        if 'A' in self.channel_name:
            self.isamp = True
            self.isphase = False
            self.istopo = False
        else:
            self.isamp = False
            if 'Z' in self.channel_name:
                self.istopo = True
                self.isphase = False
            else:
                self.istopo = False
                self.isphase = True

    def read_info_file(self, filename):
        # reader tested for neascan version 2.1.10719.0
        fid = open(filename, errors='replace')
        infodict = {}

        linestring = ''
        Nlines = 0

        while 'Version:' not in linestring:
            Nlines += 1
            linestring = fid.readline()
            if Nlines > 1:
                ct = linestring.split('\t')
                fieldname = ct[0][2:-1]
                fieldname = fieldname.replace(' ', '')

                if 'Scanner Center Position' in linestring:
                    fieldname = fieldname[:-5]
                    infodict[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Scan Area' in linestring:
                    fieldname = fieldname[:-7]
                    infodict[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Pixel Area' in linestring:
                    fieldname = fieldname[:-7]
                    infodict[fieldname] = [int(ct[2]), int(ct[3]), int(ct[4])]

                elif 'Interferometer Center/Distance' in linestring:
                    fieldname = fieldname.replace('/', '')
                    infodict[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Regulator' in linestring:
                    fieldname = fieldname[:-7]
                    infodict[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Q-Factor' in linestring:
                    fieldname = fieldname.replace('-', '')
                    infodict[fieldname] = float(ct[2])

                else:
                    fieldname = ct[0][2:-1]
                    fieldname = fieldname.replace(' ', '')
                    val = ct[2]
                    val = val.replace(',', '')
                    try:
                        infodict[fieldname] = float(val)
                    except:
                        infodict[fieldname] = val.strip()
        fid.close()
        return infodict


def line_level(inputobj: NeaImage, mtype: str):
    outputobj = copy.deepcopy(inputobj)
    match mtype:  # TODO: "match" is not python 3.9 and 3.8 compatible - we need to specify this in the requirements.
        case 'median':
            for i in range(inputobj.data.shape[0]):
                if inputobj.isamp:
                    outputobj.data[i][:] = inputobj.data[i][:]/np.median(inputobj.data[i][:])
                else:
                    outputobj.data[i][:] = inputobj.data[i][:]-np.median(inputobj.data[i][:])
        case 'average':
            for i in range(inputobj.data.shape[0]):
                if inputobj.isamp:
                    outputobj.data[i][:] = inputobj.data[i][:]/np.mean(inputobj.data[i][:])
                else:
                    outputobj.data[i][:] = inputobj.data[i][:]-np.mean(inputobj.data[i][:])
        case 'difference':
            for i in range(inputobj.data.shape[0]-1):
                if inputobj.isamp:
                    c = np.median(inputobj.data[i+1][:]/inputobj.data[i][:])
                    outputobj.data[i][:] = inputobj.data[i][:]/c
                else:
                    c = np.median(inputobj.data[i+1][:]-inputobj.data[i][:])
                    outputobj.data[i][:] = inputobj.data[i][:]-c
    return outputobj


def bg_polyfit(inputobj: NeaImage, xorder: int, yorder: int):
    outputobj = copy.deepcopy(inputobj)
    Z = copy.deepcopy(outputobj.data)
    x = list(range(0, outputobj.xres))
    y = list(range(0, outputobj.yres))
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

    basis = get_basis(x, y, xorder, yorder)
    A = np.vstack(basis).T
    b = Z.ravel()
    c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

    background = np.sum(c[:, None, None] * np.array(get_basis(X, Y, xorder, yorder)).reshape(len(basis), *X.shape),
                        axis=0)

    if inputobj.isamp:
        outputobj.data = Z/background
    else:
        outputobj.data = Z-background

    return outputobj, background


def rotate_phase(inputobj: NeaImage, degree: float):
    outputobj = copy.deepcopy(inputobj)

    # Load amplitude image
    if inputobj.isamp:
        pass
    else:
        ampIm = NeaImage()
        new_channelname = inputobj.channel_name
        new_channelname = new_channelname.replace('P', 'A')
        ampIm.read_from_gwyfile(inputobj.filename, new_channelname)
        ampIm.parameters = inputobj.parameters

        # Complex map
        C = ampIm.data * np.exp(inputobj.data*complex(1j))
        # Rotate and extract phase
        outputobj.data = np.angle(C*np.exp(np.deg2rad(degree)*complex(1j)))

    return outputobj


def self_reference(inputobj: NeaImage, order: int):
    outputobj = copy.deepcopy(inputobj)
    # Load the other harmonic
    referenceobj = NeaImage()

    channelname = ''
    if inputobj.isamp:
        channelname = f'O{order}A raw'
    elif inputobj.isphase:
        channelname = f'O{order}P raw'
    else:
        pass

    if channelname is not None:
        referenceobj.read_from_gwyfile(inputobj.filename, channelname)
        referenceobj.parameters = inputobj.parameters
    else:
        return None  # TODO: raise some error instead

    if inputobj.isamp:
        outputobj.data = np.divide(inputobj.data, referenceobj.data)
    elif inputobj.isphase:
        outputobj.data = inputobj.data-referenceobj.data
    else:
        pass

    return outputobj


def normalize_simple(inputobj: NeaImage, method='median', value=1):
    outputobj = copy.deepcopy(inputobj)
    match method:
        case 'median':
            if inputobj.isAmplitude():
                outputobj.data = inputobj.data / np.median(inputobj.data)
            else:
                outputobj.data = inputobj.data - np.median(inputobj.data)
        case 'manual':
            if inputobj.isAmplitude():
                outputobj.data = inputobj.data / value
            else:
                outputobj.data = inputobj.data - value
    return outputobj


def CalcCrossSectionRect(Rect1, Rect2):
    # TODO: implement correctly
    pass
    # x1 = Rect1(1) 
    # x2 = Rect2(1) 
    # y1 = Rect1(2) 
    # y2 = Rect2(2)
    # W1 = Rect1(3) 
    # W2 = Rect2(3) 
    # H1 = Rect1(4)
    # H2 = Rect2(4)

    # if y2 > y1:  # Positive shift
    #     Hn = H1-(y2-y1)
    #     yn = y2
    # elif y2 < y1 and y1+H1 > y2 + H2:  # Negative shift and higher than H2
    #     Hn = H2 + (y2 - y1)
    #     yn = y1
    # else:
    #     Hn = H1
    #     yn = y1