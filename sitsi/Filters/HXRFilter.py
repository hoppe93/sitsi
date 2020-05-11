"""
This video filter tries to remove pixels that are made artificially brighter
by stray X-rays.
"""

import numpy as np
import scipy.ndimage

from . Filter import Filter


class HXRFilter(Filter):
    
    
    def __init__(self, threshold=0.9):
        """
        Constructor.

        threshold: Relative amount by which a pixel must briefly change in the
                   video to be considered a HXR anomaly.
        """
        self.threshold = threshold


    def apply(self, times, data):
        """
        Apply this filter.
        """
        pixels = data.shape[1:]

        # Average in time over all pixels in the image
        time_median = np.zeros(data.shape)
        for i in range(pixels[0]):
            for j in range(pixels[1]):
                time_median[:,i,j] = scipy.ndimage.median_filter(data[:,i,j], size=3)

        # Ignore divide-by-zero for a bit
        errs = np.geterr()
        np.seterr(divide='ignore')

        # Find pixels which vary alot, but only briefly
        #   Basically, we compare the value of each pixel v(t) at times
        #   (t1, t2, t3). If v(t1)~v(t3), but v(t2) ">>" v(t1), then we
        #   interpolate the value of the pixel in time.
        tf_forward  = np.abs(time_median[:-1,:] / data[1:,:])
        tf_backward = np.abs(time_median[1:,:] / data[:-1,:])

        # Restore 'divide-by-zero' warnings
        np.seterr(**errs)

        # Locate all pixels that exceed the threshold
        # (make 'tfilter' all True to begin with)
        tfilter = (data >= 0)
        tfilter[1:-1] = (tf_forward[:-1] < self.threshold) & (tf_backward[1:] < self.threshold)

        # Interpolate anomalous pixels
        sframes = np.copy(data)
        for i in range(pixels[0]):
            for j in range(pixels[1]):
                tslice   = np.copy(data[:,i,j])
                tf_slice = tfilter[:,i,j]

                # In every time point where this pixel is anomalously bright,
                # we interpolate its value based on its previous and next (in time) value
                tslice[tf_slice] = np.interp(times[tf_slice], times[~tf_slice], tslice[~tf_slice])
                sframes[:,i,j] = tslice

        return sframes


