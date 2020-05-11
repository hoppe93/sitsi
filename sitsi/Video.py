
import h5py
import numpy as np

from .Image import Image
from .InverterException import InverterException


class Video:
    
    def __init__(self, data=None, filters=list()):
        """
        Constructor.

        data: May be either a filename (in which case the data is loaded
              from the named file) or a Video object which should be copied.
        """
        self.frames    = list()
        self.rawframes = list()
        self.times     = list()
        self.framemax  = 0
        self.info      = dict()
        self.X         = list()
        self.Y         = list()

        self.filters  = filters

        self.true_framemaxs = [None]*len(self.frames)

        if type(data) == str:
            self.loadVideoHDF5(data)
        elif type(data) == Video:
            self.frames    = np.copy(data.frames)
            self.rawframes = np.copy(data.rawframes)
            self.times     = np.copy(data.times)
            self.framemax  = data.framemax
            self.X         = np.copy(data.X)
            self.Y         = np.copy(data.Y)
            self.info      = np.copy(data.info)

            self.true_framemaxs = np.copy(data.true_framemaxs)
        else:
            raise InverterException("Unrecognized type of 'data' input.")

        self.applyFilters()

    
    def applyFilters(self):
        """
        Apply all the filters assigned to this video.
        """
        self.rawframes = self.frames
        d = np.copy(self.frames)

        for f in self.filters:
            d = f.apply(self.times, d)

        self.frames = d


    def computeTrueMaxima(self):
        for i in range(0, len(self.frames)):
            self.getTrueMaximum(i)

        return self.true_framemaxs


    def loadVideoHDF5(self, filename):
        """
        Loads the video from the file with the given name.

        filename: Name of file to load.
        """
        with h5py.File(filename, 'r') as f:
            self.frames    = f['frames'][:].transpose((0,2,1)).astype(np.double)
            self.rawframes = np.copy(self.frames)
            self.times     = f['times'][:]
            self.framemax  = np.amax(self.frames)

            self.X = list(range(self.frames[0].shape[0]-1, -1, -1))
            self.Y = list(range(0, self.frames[0].shape[1]))

            self.true_framemaxs = [None]*len(self.frames)

            self.info = dict()
            for key in f['info'].keys():
                self.info[key] = f['info'][key][:]


    def getFrame(self, frameindex, frm=None):
        """
        Returns the frame with the given index as an 'Image' object.
        """
        if frm is None:
            frm = self.frames

        img = Image(frm[frameindex])
        img.setSubset(*self.subset)

        return img


    def getFrameUnfiltered(self, frameindex):
        """
        Returns the frame with the given index as an 'Image' object,
        without applying any filters.
        """
        return self.getFrame(frameindex, frm=self.rawframes)


    def getTrueMaximum(self, frameindex, threshold=1e-3, order=5):
        """
        Returns the 'true' maximum of a video frame. This function is
        intended to return the maximum intensity of any pixel in the
        image, with HXR-saturated pixels discarded.

        frame:     Index of video frame to get true maximum of.
        threshold: Threshold for considering pixels to "lie close".
        order:     Order of method (number of relative differences to compute).
        """
        if self.true_framemaxs[frameindex] is not None:
            return self.true_framemaxs[frameindex]

        f = np.sort(self.frames[frameindex], axis=None)

        i = f.size-1
        r = lambda index : np.abs(f[index] - f[index-1]) / f[index]

        rv = np.zeros((order,))
        for j in range(0, order): rv[j] = r(i-j)

        while i > 0 and np.any(rv > threshold):
            i -= 1
            for j in range(0, order): rv[j] = r(i-j)

        self.true_framemaxs[frameindex] = f[i]

        return f[i]
    

    def interpolate(self, times):
        """
        Interpolate the frames of this video to
        the times in the given list. A 'closest'
        interpolation is done, meaning that the
        frame corresponding to the closest time of
        each element in 'times' is selected.
        
        times: List of times to interpolate this video to.
        """
        nframes = list()
        for i in range(0, len(times)):
            j = np.argmin(np.abs(self.times - times[i]))
            nframes.append(self.frames[j])

        self.frames = np.array(nframes)
        self.times  = np.array(times)
        self.framemax = np.amax(self.frames)


    def setSubset(self, x, y=None, w=None, h=None):
        """
        Specifies a subset of each video frame which will be applied 
        to all frames returned by 'getFrame()'. Calling this method
        like 'setSubet(None)' resets any previously specified subset.
        """
        self.subset = (x,y,w,h)


