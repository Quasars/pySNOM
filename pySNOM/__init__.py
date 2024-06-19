"""
Module pySNOM
=============
Scanning Near-field Optical Microscopy (SNOM) analysis tools
"""

from .NeaImager import NeaImage
from .NeaSpectra import NeaSpectrum

__version__ = '0.0.2'
__author__ = 'Gergely Németh, Ferenc Borondics'
__credits__ = 'Wigner Research Centre for Physics, Synchrotron SOLEIL'
__all__ = ["NeaImage", "NeaSpectrum"]
