"""
Module pySNOM
=============
Scanning Near-field Optical Microscopy (SNOM) analysis tools
"""
from .defaults import defaults
from .images import Image, Transformation
from .readers import Reader
from .spectra import NeaSpectrum
# import images
# import spectra
# import readers

__version__ = '0.0.2'
__author__ = 'Gergely Németh, Ferenc Borondics'
__credits__ = 'Wigner Research Centre for Physics, Synchrotron SOLEIL'
__all__ = ["defaults","Reader","Image","Transformation"]
