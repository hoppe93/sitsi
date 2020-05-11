
import numpy as np
from . InverterException import InverterException
from . InputData import InputData


class Image(InputData):
    """
    This class represents a camera image and can be used as input to the
    various inversion algorithms. Images can be created directly, or by
    importing and filtering a video.
    """
    

    def __init__(self, data):
        """
        Constructor.

        Args:
            data (numpy.ndarray): Raw image data, or Image object to copy.
        """
        if data.ndim != 2:
            raise InverterException("Invalid dimensions of image: {}. Image must have exactly two dimensions.".format(data.ndim))

        self.data   = data
        self.pixels = data.shape
        self.subset = (slice(None), slice(None))


    def get(self):
        """
        Returns:
            numpy.ndarray: the image data, or the previously specified subset of the image data.
        """
        return self.data[self.subset]


    def setSubset(self, x, y=None, w=None, h=None):
        """
        Specifies which subset of the image to return when
        'get()' is called. Calling this method as 'setSubset(None)'
        resets any previously set subset.

        Args:
            x (int): X axis offset.
            y (int): Y axis offset.
            w (int): Number of pixels to pick along X axis.
            h (int): Number of pixels to pick along Y axis.
        """
        if (x is None) and (y is None) and (w is None) and (h is None):
            self.subset = (slice(None), slice(None))
        else:
            self.subset = (slice(x, x+w), slice(y, y+h))


