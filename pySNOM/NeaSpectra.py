import numpy as np
import os
import copy
from scipy import signal
from scipy.fft import fft, fftshift
from scipy.interpolate import CubicSpline, interp1d
import copy

class NeaSpectrum:
    def __init__(self) -> None:
        self.filename = None # Full path with name
        # data from all the channels
        self.data = {}
        # Other parameters from info txt -  Dictionary
        self.parameters = {}

    def readNeaSpectrum(self,filename):
        # reader tested for neascan version 2.1.10719.0
        self.filename = filename
        fid = open(filename,errors='replace')
        data = {}
        params = {}

        linestring = fid.readline()
        Nlines = 1

        while 'Row' not in linestring:
            Nlines += 1
            linestring = fid.readline()
            if Nlines > 1:
                ct = linestring.split('\t')
                fieldname = ct[0][2:-1]
                fieldname = fieldname.replace(' ', '')

                if 'Scanner Center Position' in linestring:
                    fieldname = fieldname[:-5]
                    params[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Scan Area' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Pixel Area' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [int(ct[2]), int(ct[3]), int(ct[4])]

                elif 'Interferometer Center/Distance' in linestring:
                    fieldname = fieldname.replace('/', '')
                    params[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Regulator' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Q-Factor' in linestring:
                    fieldname = fieldname.replace('-', '')
                    params[fieldname] = float(ct[2])

                else:
                    fieldname = ct[0][2:-1]
                    fieldname = fieldname.replace(' ', '')
                    val = ct[2]
                    val = val.replace(',','')
                    try:
                        params[fieldname] = float(val)
                    except:
                        params[fieldname] = val.strip()

        channels = linestring.split('\t')
        self.parameters = params
        fid.close()

        if "PTE+" in params['Scan']:
            C_data = np.genfromtxt(filename, skip_header=Nlines, encoding='utf-8')
        else:
            C_data = np.genfromtxt(filename, skip_header=Nlines)

        for i in range(len(channels)-2):
            if params['PixelArea'][1] == 1 and params['PixelArea'][0] == 1:
                if "PTE+" in params['Scan']:
                    data[channels[i]] = np.reshape(C_data[:,i], (params['PixelArea'][2]))
                else:
                    data[channels[i]] = np.reshape(C_data[:,i], (params['PixelArea'][2]*2))
            else:
                if "PTE+" in params['Scan']:
                    data[channels[i]] = np.reshape(C_data[:,i], (params['PixelArea'][0], params['PixelArea'][1], params['PixelArea'][2]))
                else:
                    data[channels[i]] = np.reshape(C_data[:,i], (params['PixelArea'][0], params['PixelArea'][1], params['PixelArea'][2]*2))
        self.data = data

    def saveSpectraToDAT(self,channelname):
        fname = f'{self.filename[0:-4]}.dat'
        M = np.array([self.data["Wavenumber"],self.data[channelname]])
        np.savetxt(fname, M.T)

    def linearSubtract(self,channelname,wn1,wn2):        
        if self.parameters['PixelArea'][0] == 1 and self.parameters['PixelArea'][1] == 1:
            wnaxis = self.data["Wavenumber"]
            data = self.data[channelname]
            wn1idx = np.argmin(abs(wnaxis-wn1))
            wn2idx = np.argmin(abs(wnaxis-wn2))
            m = (data[wn2idx]-data[wn1idx])/(wnaxis[wn2idx]-wnaxis[wn1idx])
            C = data[wn1idx]-m*wnaxis[wn1idx]

            self.data[channelname] = data - (m*wnaxis + C)
        else:
            for i in range(self.parameters['PixelArea'][0]):
                    for k in range(self.parameters['PixelArea'][1]):
                        wnaxis = self.data["Wavenumber"][i,k,:]
                        data = self.data[channelname][i,k,:]
                        wn1idx = np.argmin(abs(wnaxis-wn1))
                        wn2idx = np.argmin(abs(wnaxis-wn2))
                        m = (data[wn2idx]-data[wn1idx])/(wnaxis[wn2idx]-wnaxis[wn1idx])
                        C = data[wn1idx]-m*wnaxis[wn1idx]

                        self.data[channelname][i,k,:] = data - (m*wnaxis + C)

    def normalizeSpectrum(self, ref_spectrum, order = None, dounwrap = "True"):
        # if order is not defined, go trough all the orders
        if order == None:
            for i in [0,1,2,3,4,5]:
                channelA = f"O{i}A"
                channelP = f"O{i}P"
                for i in range(self.parameters['PixelArea'][0]):
                    for k in range(self.parameters['PixelArea'][1]):

                        self.data[channelA][i,k,:] = self.data[channelA][i,k,:]/ref_spectrum.data[channelA]
                        self.data[channelP][i,k,:] = self.data[channelP][i,k,:]-ref_spectrum.data[channelP]

                        if dounwrap:
                            self.data[channelP][i,k,:] = np.unwrap(self.data[channelP][i,k,:])

        else:
            channelA = f"O{order}A"
            channelP = f"O{order}P"
            # Check if this is juts a point spectrum
            if self.parameters['PixelArea'][0] == 1 and self.parameters['PixelArea'][1] == 1:

                self.data[channelA] = self.data[channelA]/ref_spectrum.data[channelA]
                self.data[channelP] = self.data[channelP]-ref_spectrum.data[channelP]

                if dounwrap:
                    self.data[channelP] = np.unwrap(self.data[channelP])
            else:
                for i in range(self.parameters['PixelArea'][0]):
                    for k in range(self.parameters['PixelArea'][1]):

                        self.data[channelA][i,k,:] = self.data[channelA][i,k,:]/ref_spectrum.data[channelA]
                        self.data[channelP][i,k,:] = self.data[channelP][i,k,:]-ref_spectrum.data[channelP]

                        if dounwrap:
                            self.data[channelP][i,k,:] = np.unwrap(self.data[channelP][i,k,:])

    def reshapeForImshow(self):
        plotdata = np.reshape(np.ravel(self.data["O2A"]),(self.parameters["PixelArea"][0],self.parameters["PixelArea"][2]))
        return plotdata

class NeaInterferogram:
    def __init__(self) -> None:
        self.filename = None # Full path with name
        # data from all the channels
        self.data = {}
        # Other parameters from info txt -  Dictionary
        self.parameters = {}

    def readNeaInterferogram(self,filename):
        # reader tested for neascan version 2.1.10719.0
        self.filename = filename
        fid = open(filename,errors='replace')
        data = {}
        params = {}

        linestring = fid.readline()
        Nlines = 1

        while 'Row' not in linestring:
            Nlines += 1
            linestring = fid.readline()
            if Nlines > 1:
                ct = linestring.split('\t')
                fieldname = ct[0][2:-1]
                fieldname = fieldname.replace(' ', '')

                if 'Scanner Center Position' in linestring:
                    fieldname = fieldname[:-5]
                    params[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Scan Area' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Pixel Area' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [int(ct[2]), int(ct[3]), int(ct[4])]

                elif 'Averaging' in linestring:
                    # fieldname = fieldname[:-7]
                    params[fieldname] = int(ct[2])

                elif 'Interferometer Center/Distance' in linestring:
                    fieldname = fieldname.replace('/', '')
                    params[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Regulator' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Q-Factor' in linestring:
                    fieldname = fieldname.replace('-', '')
                    params[fieldname] = float(ct[2])

                else:
                    fieldname = ct[0][2:-1]
                    fieldname = fieldname.replace(' ', '')
                    val = ct[2]
                    val = val.replace(',','')
                    try:
                        params[fieldname] = float(val)
                    except:
                        params[fieldname] = val.strip()

        channels = linestring.split('\t')
        self.parameters = params
        fid.close()

        C_data = np.genfromtxt(filename, skip_header=Nlines)

        for i in range(len(channels)-2):
            if params['PixelArea'][1] == 1 and params['PixelArea'][0] == 1:
                data[channels[i]] = C_data[:,i]
            else:
                data[channels[i]] = np.reshape(C_data[:,i], (int(params['PixelArea'][0]), int(params['PixelArea'][1]), int(params['PixelArea'][2]*params['Averaging'])))
        self.data = data

    def reshapeSinglePointFromChannel(self,channel):
        interferograms = self.data[channel]
        # TODO error handling if params['PixelArea'][1] or params['PixelArea'][0] =! 1
        return np.reshape(interferograms,(int(self.parameters['Averaging']),self.parameters['PixelArea'][2]))
    
    def asymmetricWindow(self, npoints = 2048, centerindex = 1700, windowtype = "blackmanharris"):
        """
        Calculates an asymmetric blackmann-harris window function with length given by *npoints*.
        The center of the window is defined by *centerindex*

        Parameters
        ----------
        npoints : int
            Length of the final window function
        centerindex : int
            Location index of the maximum of the window function.
            If `n` is smaller than the length of the input, the input is cropped.
            If it is larger, the input is padded with zeros. If `n` is not given,
            the length of the input along the axis specified by `axis` is used.

        Returns
        -------
        out : real-valued ndarray of the window function

        Raises
        ------

        See Also
        --------

        Notes
        -----

        References
        ----------

        Examples
        --------

        """
        # Calculate the length of the two sides
        length1 = (centerindex)*2
        length2 = (npoints-centerindex)*2

        # Generate the two parts of the window

        windowfunc = getattr(signal.windows,windowtype)
        windowFull1 = windowfunc(length1)
        windowFull2 = windowfunc(length2)

        # Construct the asymetric window from the two sides
        asymWindow1 = windowFull1[0:int(len(windowFull1)/2)]
        asymWindow2 = windowFull2[int(len(windowFull2)/2):int(len(windowFull2))]
        asymWindowFull = np.concatenate((asymWindow1, asymWindow2))

        return asymWindowFull
    
    def processSingleInterferogram(self, ifg, maxis, windowtype = "blackmanharris", wlpindex = 1024, nzeros = 4, apod = True, autoidx = True):
        # Find the location index of the WLP
        if autoidx:
            wlpindex = np.argmax(np.abs(ifg))
        else:
            pass
        # Create apodization window
        if apod:
            w = self.asymmetricWindow(npoints = len(ifg), centerindex = wlpindex, windowtype = windowtype)
        else:
            w = np.ones(np.shape(ifg))

        # if np.iscomplex(ifg).any():
        #     real_ifg = np.real(ifg)
        #     imag_ifg = np.imag(ifg)
        #     real_ifg = real_ifg - np.mean(real_ifg)
        #     imag_ifg = imag_ifg - np.mean(imag_ifg)
        #     ifg = real_ifg + imag_ifg*complex(1j)
        # else:
        ifg = ifg - np.mean(ifg)

        # Calculate FFT
        complex_spectrum = fftshift(fft(ifg*w, nzeros*len(ifg)))
        # amplitude = np.abs(complex_spectrum)
        # phase = np.angle(complex_spectrum)

        # Calculate frequency axis
        stepsizes = np.mean(np.diff(maxis*1e6))
        Fs = 1/np.mean(stepsizes)
        faxis = (Fs/2)*np.linspace(-1,1,len(complex_spectrum))*10000/2

        # return amplitude[int(len(faxis)/2)-1:-1], phase[int(len(faxis)/2)-1:-1], faxis[int(len(faxis)/2)-1:-1]
        return complex_spectrum[int(len(faxis)/2)-1:-1], faxis[int(len(faxis)/2)-1:-1]
    
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