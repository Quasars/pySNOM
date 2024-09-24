import numpy as np
import os
import copy
from enum import Enum
from scipy import signal
from scipy.fft import fft, fftshift
from scipy.interpolate import CubicSpline, interp1d
import copy

MeasurementModes = Enum('MeasurementModes', ['None','nanoFTIR'])
DataTypes = Enum('DataTypes', ['Amplitude', 'Phase', 'Topography'])
ChannelTypes = Enum('ChannelTypes', ['None','Optical','Mechanical'])
ScanTypes = Enum('ScanTypes',['Point','LineScan','HyperScan'])

# INTERFEROGRAMS ------------------------------------------------------------------------------------------------------------------
class NeaInterferogram:
    def __init__(self) -> None:
        self.filename = None # Full path with name
        # data from all the channels
        self.data = {}
        # Other parameters from info txt -  Dictionary
        self.parameters = {}

    
    def processPointInterferogram(self, simpleoutput = False, order = 2, windowtype = "blackmanharris", nzeros = 4, apod = True, method = "complex", interpmethod = "spline"):
        
        # Load amplitude and phase of the given channel
        channelA = f"O{order}A"
        channelP = f"O{order}P"

        ifgA = self.reshapeSinglePointFromChannel(channelA)
        ifgP = self.reshapeSinglePointFromChannel(channelP)

        # Load position data
        Maxis = self.reshapeSinglePointFromChannel("M")

        # Calculate the interferogram to process based on the given method
        match method:
            case "abs":
                IFG = np.abs(ifgA*np.exp(ifgP*complex(1j)))
            case "real":
                IFG = np.real(ifgA*np.exp(ifgP*complex(1j)))
            case "imag":
                IFG = np.imag(ifgA*np.exp(ifgP*complex(1j)))
            case "complex":
                IFG = ifgA*np.exp(ifgP*complex(1j))
            case "simple":
                IFG = ifgA
        #  Interpolate
        match method:
            case "complex":
                IFG, Maxis = self.interpolateSingleInterferogram(IFG,Maxis,method = interpmethod)
                # realIFG, Maxis = self.interpolateSingleInterferogram(np.real(IFG),Maxis,method = interpmethod)
                # imagIFG, Maxis = self.interpolateSingleInterferogram(np.imag(IFG),Maxis,method = interpmethod)
                # IFG = realIFG + complex(1j)*imagIFG
            case _:
                IFG, Maxis = self.interpolateSingleInterferogram(IFG,Maxis,method = interpmethod)
        
        # PROCESS IFGs
        # Check if it is multiple interferograms or just a single one
        if len(np.shape(IFG)) == 1:
            complex_spectrum, f = self.processSingleInterferogram(IFG, Maxis, windowtype = windowtype, nzeros = nzeros, apod = apod, autoidx = True)
            amp = np.abs(complex_spectrum)
            phi = np.angle(complex_spectrum)
        else:
            # Allocate variables
            spectraAll = complex(1j)*np.zeros((np.shape(IFG)[0],int(nzeros*np.shape(IFG)[1]/2)))
            fAll = np.zeros(np.shape(spectraAll))
            # Go trough all
            for i in range(np.shape(IFG)[0]):
                spectraAll[i,:], fAll[i,:] = self.processSingleInterferogram(IFG[i,:], Maxis[i,:], windowtype = windowtype, nzeros = nzeros, apod = apod, autoidx = True)
            # Average the complex spectra
            complex_spectrum = np.mean(spectraAll, axis = 0)
            # Extract amplitude and phase from the averaged complex spectrum
            amp = np.abs(complex_spectrum)
            phi = np.angle(complex_spectrum)
            f = np.mean(fAll, axis=0)

        if simpleoutput:
            return amp, phi, f
        else:
            spectrum = NeaSpectrum()
            spectrum.parameters = copy.deepcopy(self.parameters)
            spectrum.parameters["ScanArea"] = [self.parameters["ScanArea"][0],self.parameters["ScanArea"][1],len(amp)]
            spectrum.data[channelA] = amp
            spectrum.data[channelP] = phi
            spectrum.data["Wavenumber"] = f

            return spectrum

    def processAllPoints(self, order = 2, windowtype = "blackmanharris", nzeros = 4, apod = True, method = "complex", interpmethod = "spline"):
        if self.parameters['PixelArea'][0] == 1 and self.parameters['PixelArea'][1] == 1:
            spectra = self.processPointInterferogram(order = order, windowtype = windowtype, nzeros = nzeros, apod = apod, method = method, interpmethod = interpmethod)
        else:
            spectra = NeaSpectrum()
            spectra.parameters = copy.deepcopy(self.parameters)
            spectra.parameters["PixelArea"] = [self.parameters["PixelArea"][0],self.parameters["PixelArea"][1],int(nzeros*self.parameters['PixelArea'][2]/2)]

            ampFullData = np.zeros((spectra.parameters['PixelArea'][0], spectra.parameters['PixelArea'][1],spectra.parameters["PixelArea"][2]))
            phiFullData = np.zeros(np.shape(ampFullData))
            fFullData = np.zeros(np.shape(ampFullData))

            singlePointIFG = NeaInterferogram()
            singlePointIFG.parameters = copy.deepcopy(self.parameters)
            singlePointIFG.parameters["PixelArea"] = [1,1,self.parameters["PixelArea"][2]]

            singlePointIFG.data = dict()

            channelA = f"O{order}A"
            channelP = f"O{order}P"
            
            for i in range(self.parameters['PixelArea'][0]):
                for k in range(self.parameters['PixelArea'][1]):
                    singlePointIFG.data[channelA] = self.data[channelA][i,k,:]
                    singlePointIFG.data[channelP] = self.data[channelP][i,k,:]
                    singlePointIFG.data["M"] = self.data["M"][i,k,:]
                    ampFullData[i,k,:], phiFullData[i,k,:], fFullData[i,k,:] = singlePointIFG.processPointInterferogram(order = order, simpleoutput = True, windowtype = windowtype, nzeros = nzeros, apod = apod, method = method, interpmethod = interpmethod)
            
            spectra.data[channelA] = ampFullData
            spectra.data[channelP] = phiFullData
            spectra.data["Wavenumber"] = fFullData

        return spectra

    def interpolateSingleInterferogram(self, ifg, maxis, method = "spline"):
        """
        Re-interpolates the an interferogram to have a uniform coordinate spacing. 
        First, recalculates the space coordinates from sampling coordinates givan by *maxis*. 
        The interferogram is then re-interpolated for the new space grid.
        
        Parameters
        ----------
        ifg : 2d array or list
            Containing the interferograms row-wise (first index)
        maxis : 2d array or list
            Containing the position coordinates for each interferograms row-wise (first index)

        """
        if np.iscomplex(ifg).any():
            newifg = np.zeros(np.shape(ifg))*complex(1j)
        else:
            newifg = np.zeros(np.shape(ifg))
        
        newmaxis = np.zeros(np.shape(maxis))

        # startM = np.min(maxis)
        # stopM = np.max(maxis)

        startM = np.min(np.median(maxis,axis=0))
        stopM = np.max(np.median(maxis,axis=0))

        try:
            newcoords = np.linspace(startM,stopM,num=np.shape(maxis)[1])
            for i in range(np.shape(ifg)[0]):
                spl = CubicSpline(maxis[i][:], ifg[i][:])
                newifg[i][:] = spl(newcoords)
                newmaxis[i][:] = newcoords
        except:
            newcoords = np.linspace(startM,stopM,num = len(maxis))
            match method:
                case "spline":
                    interp_object = CubicSpline(maxis, ifg)
                    newifg = interp_object(newcoords)
                case "linear":
                    interp_object = interp1d(maxis, ifg)
                    newifg = interp_object(newcoords)

            newmaxis = newcoords

        return newifg, newmaxis

    def analyseRealSteps(self, maxis):
        # maxis = self.reshapeSinglePointFromChannel("M")*1e6
        stepsize = np.zeros((np.shape(maxis)[0],1))
        stepspread = np.zeros((np.shape(maxis)[0],1))
        for i in range(np.shape(maxis)[0]):
            stepsize[i] = np.mean(np.diff(maxis[i,:]))
            stepspread[i] = np.std(np.diff(maxis[i,:]))

        return stepsize, stepspread

# TRANSFORMATIONS ------------------------------------------------------------------------------------------------------------------
class Transformation:

    def transform(self, data):
        raise NotImplementedError()
    
class LinearNormalize(Transformation):

    def __init__(self, wavenumber1=0.0, wavenumber2=1000.0, datatype=DataTypes.Phase):
        pass

    def transform(self, spectrum, wnaxis):
        pass
class ProcessInterferogram(Transformation):

    def __init__(self, apod = True, nzeros = 4, wlpidx = None):
        self.apod = apod
        self.nzeros = nzeros
        self.wlpidx = wlpidx

    def transform(self, ifg, maxis, windowtype = "blackmanharris"):
        # Find the location index of the WLP
        if self.wlpidx is None:
            self.wlpidx = np.argmax(np.abs(ifg))

        # Create apodization window
        if self.apod:
            w = Tools.asymmetric_window(npoints = len(ifg), centerindex = self.wlpidx, windowtype = windowtype)
        else:
            w = np.ones(np.shape(ifg))

        ifg = ifg - np.mean(ifg)

        # Calculate FFT
        complex_spectrum = fftshift(fft(ifg*w, self.nzeros*len(ifg)))

        # Calculate frequency axis
        stepsizes = np.mean(np.diff(maxis*1e6))
        Fs = 1/np.mean(stepsizes)
        faxis = (Fs/2)*np.linspace(-1,1,len(complex_spectrum))*10000/2

        # return amplitude[int(len(faxis)/2)-1:-1], phase[int(len(faxis)/2)-1:-1], faxis[int(len(faxis)/2)-1:-1]
        return complex_spectrum[int(len(faxis)/2)-1:-1], faxis[int(len(faxis)/2)-1:-1]


# TOOLS ------------------------------------------------------------------------------------------------------------------
class Tools:
    def __init__(self):
        pass

    def reshape_linescan(data, parameters):
        return np.reshape(np.ravel(data),(parameters["PixelArea"][0],parameters["PixelArea"][2]))
    
    def reshape_interferograms(data, parameters):
        # TODO error handling if params['PixelArea'][1] or params['PixelArea'][0] =! 1
        return np.reshape(data,(int(parameters['Averaging']),parameters['PixelArea'][2]))
    
    def asymmetric_window(self, npoints, centerindex=None, windowtype="blackmanharris"):
        
        if centerindex is None:
            centerindex = int(len(windowPart2)/2)

        # Calculate the length of the two sides
        length1 = (centerindex)*2
        length2 = (npoints-centerindex)*2

        # Generate the two parts of the window

        windowfunc = getattr(signal.windows,windowtype)
        windowPart1 = windowfunc(length1)
        windowPart2 = windowfunc(length2)

        # Construct the asymetric window from the two sides
        asymWindow1 = windowPart1[0:int(len(windowPart1)/2)]
        if npoints % 2 == 0:
            asymWindow2 = windowPart2[int(len(windowPart2)/2):int(len(windowPart2))]
        else:
            asymWindow2 = windowPart2[int(len(windowPart2)/2+1):int(len(windowPart2))]

        return np.concatenate((asymWindow1, asymWindow2))