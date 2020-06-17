
import numpy as np
from . InverterException import InverterException
from . InputData import InputData


class Spectrum(InputData):
    """
    This class represents a power spectrum and can be used as input to the
    various inversion algorithms.
    """


    def __init__(self, data, wavelengths=None):
        """
        Constructor.

        Args:
            data (numpy.ndarray):        Raw spectrum data, or Spectrum
                                         object to copy.
            wavelengths (numpy.ndarray): Wavelength vector.
        """
        if data.ndim != 1:
            raise InverterException("Invalid dimensions of spectrum: {}. Spectrum must have exactly one dimension.".format(data.ndim))

        if type(data) == Spectrum:
            self.data        = data.data
            self.wavelengths = data.wavelengths
        else:
            self.data        = data
            self.wavelengths = wavelengths


        if self.wavelengths is None:
            raise InverterException("Wavelengths vector must be given.")


    def get(self):
        """
        Returns:
            numpy.ndarray: the spectrum data.
        """
        return self.data


