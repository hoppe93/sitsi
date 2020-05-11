# General image filter base class


import numpy as np
from .. Video import Video


class Filter:
    

    def __init__(self):
        """
        Constructor.
        """
        pass

    
    def apply(self, times, data):
        """
        This method should be overridden by derived filter classes.
        It should construct a new 'self.frames' array based on the
        'self.rawframes' array.

        This method should also return the processed data.
        """
        self.frames = data
        return self.frames


