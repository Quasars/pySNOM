""" This file contains a dictionary of definitions to create connection
between the manufacturer nomeclature and the naming used in the package """


class Defaults:
    # Neaspec names are taken from the Scan field of the info txt file
    def __init__(self) -> None:
        self.image_mode_defs = {
            "AFM": "AFM",
            "2D (PsHet)": "PsHet",
            "Whitelight Imaging": "WLI",
            "Photo Thermal Expansion+": "PTE",
            "Tapping AFM-IR+": "TappingAFMIR",
            "Contact Mode 2D": "ContactAFM",
        }

        self.spectral_mode_defs = {
            "Fourier Scan": "nanoFTIR",
            "Pointspectroscopy PTE+": "PTE",
            "AFM-Raman/PL Scan (Tapping Mode)": "nanoRaman",
        }


defaults = Defaults()
