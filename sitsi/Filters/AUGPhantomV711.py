
import numpy as np
from . Filter import Filter
from . HXRFilter import HXRFilter
from . RemoveArtifacts import RemoveArtifacts


class AUGPhantomV711(Filter):
    

    def __init__(self):
        """
        Constructor.
        """
        self.filters = []
        self.filters.append(RemoveArtifacts())
        self.filters.append(HXRFilter())


    def apply(self, times, data):
        """
        Filters the raw video data so as to remove the noise
        found in ASDEX-U Phantom V711 camera images.
        """
        d = np.copy(data)
        for f in self.filters:
            d = f.apply(times=times, data=d)

        return d


