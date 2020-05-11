"""
This video filter removes fixed structures in an image, i.e. features in a
camera image which are not physical, and are encoded in the camera hardware.
This filter was invented to remove repetitive and clearly artifical patterns
in camera images from the Phantom V711 visible light camera at ASDEX-U.
"""

import numpy as np
import scipy.ndimage

from . Filter import Filter


class RemoveArtifacts(Filter):
    
    
    def __init__(self, sigma=0.8):
        self.sigma = sigma


    def apply(self, times, data):
        """
        Apply this filter.
        """
        mean       = np.mean(data, axis=0)
        smoothed   = scipy.ndimage.gaussian_filter(mean, self.sigma)

        return data * (smoothed/mean)

