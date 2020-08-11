"""
This represents a large Green's function which has been split across multiple
files.
"""

import h5py
import numpy as np

from . InverterException import InverterException


class SuperGreensFunction:
    

    def __init__(self, files, splitdim=0):
        """
        Constructor.

        files:    List of files containing the Green's function. If the list
                  contains multiple files, it is assumed that Green's function
                  is split along one dimension into the separate files.
        splitdim: Index of dimension which has been split. Alternatively, a
                  string specifying the name of the dimension can given.
        """
        self.format = None
        self.paramlist = {}

        # List which allows us to map Green's function parameter indices
        # to Green's function file names
        self.fileparamlist = []

        # This list allows us to map Green's function parameter indices in
        # the super Green's function, to parameter indices in a single (/local)
        # Green's function
        self.localindices  = []

        self.r  = None
        self.p1 = None
        self.p2 = None
        self.w  = None
        self.pixels = (0, 0)

        self.processfile(files, splitdim=splitdim)


    def processfile(self, files, splitdim):
        """
        Process the list of input Green's function files in order to get
        a picture of what the full function actually consists of.
        """
        if files[0].endswith('.mat'):
            tos = lambda v : "".join(map(chr, v[:,:][:,0].tolist()))
        else:
            tos = lambda v : v[:].tostring().decode('utf-8')

        for f in files:
            with h5py.File(f, 'r') as fh:
                if self.format is None:
                    self.format = tos(fh['type'])

                    if type(splitdim) == int:
                        self.splitdim = splitdim
                    else:
                        self.splitdim = self.format.find(splitdim)
                        if self.splitdim < 0:
                            raise InverterException("Invalid split dimension specified.")

                    self.param1name = tos(fh['param1name'])
                    self.param2name = tos(fh['param2name'])

                    # We assume that the super Green's function is split based
                    # on the first dimension (which may not be a pixel).
                    #if (f0 != 'r') and (f0 != '1') and (f0 != '2') and (f0 != 'w'):
                    #    raise InverterException("Invalid first dimension of Green's function.")
                else:   # Verify that the format is the same as for all other files
                    if self.format != tos(fh['type']):
                        raise InverterException("The specified Green's functions have different formats.")

                    if (self.param1name != tos(fh['param1name'])) or (self.param2name != tos(fh['param2name'])):
                        raise InverterException("The specified Green's functions have different momentum parameters.")

                r  = fh['r'][:]
                p1 = fh['param1'][:]
                p2 = fh['param2'][:]
                w  = fh['wavelengths'][:]

                if 'rowpixels' in fh:
                    self.pixels = (int(fh['rowpixels'][:][0]), int(fh['colpixels'][:][0]))

                self._setParameters(r, p1, p2, w, filename=f)


        # Convert the constructed parameter list to a list of indices
        fileparamlist = []
        indices = []
        localindices = []
        for key, val in self.paramlist.items():
            l = list()
            i = 0
            for v in val:
                indices.append(self.getParameterIndex(v))
                fileparamlist.append(key)
                localindices.append(i)

                i += 1

        self.fileparamlist = [f for _, f in sorted(zip(indices, fileparamlist))]
        self.localindices  = localindices


    def __getitem__(self, index):
        """
        Retrieve the Green's function value by index. The first index is
        the varied parameter. If it is a float, the data for the function
        with a parameter value closest to the one specified is returned.
        """
        p = index[self.splitdim]
        f = None

        if type(p) == int or type(p) == tuple:
            f = self._getFunction(p)
        elif type(p) == slice:
            f = self._getFunction(index)
        elif type(p) == float:
            f = self._getFunction(self.getParameterIndex(p))
        else:
            raise InverterException("Invalid type '{}' of first index.".format(type(p)))

        if len(index) > 1:
            return f[index[1:]]
        else:
            return f


    def get(self, r=None, p1=None, p2=None, w=None, i=None, j=None):
        """
        This function allows you to extract a portion of the Green's
        function based on parameter names instead of index locations.
        The arguments can either be ranges or 'None', the latter
        being the same as specifying ':' (i.e. all elements).
        """
        return self._getFunction(self.getSlice(r=r, p1=p1, p2=p2, w=w, i=i, j=j))


    def _getFunction(self, idx):
        """
        Returns the Green's function with the specified index.
        """
        filename = None

        if type(idx) == tuple:
            filename = self.fileparamlist[idx[self.splitdim]]
            lst = []

            # Get the selection as a list of indices (or range)
            if type(idx[self.splitdim]) == slice:
                ix = idx[self.splitdim]
                ifnone = lambda a, b: b if a is None else a
                lst = range(ifnone(ix.start, 0), ifnone(ix.stop, len(self.fileparamlist)), ifnone(ix.step, 1))
                filename = filename[0]
            else:
                lst = [idx[self.splitdim]]

            # Check that all elements of split dimension are in
            # the same physical file
            for i in lst:
                if filename != self.fileparamlist[i]:
                    raise InverterException("Trying to load multiple Green's functions simultaneously.")
        elif type(idx) == int:
            filename = self.fileparamlist[idx]

        func = None
        with h5py.File(filename, 'r') as f:
            func = f['func'][:]

        if func.ndim == len(self.format):
            ix = list(idx)
            ix[self.splitdim] = self.localindices[idx[self.splitdim]]
            return func[tuple(ix)]
        else:
            return func


    def getParameterIndex(self, v):
        """
        Returns the index of the varied parameter corresponding to the
        given parameter value.
        """
        arr = None
        if   self.format[self.splitdim] == 'r': arr = self.r
        elif self.format[self.splitdim] == '1': arr = self.p1
        elif self.format[self.splitdim] == '2': arr = self.p2
        elif self.format[self.splitdim] == 'w': arr = self.w
            
        return np.argmin(np.abs(arr-v))


    def getSlice(self, r=None, p1=None, p2=None, w=None, i=None, j=None):
        """
        Returns a tuple of slices which will extract the specified
        parts of the Green's function when used as an index.
        """
        fmt = self.format
        l = len(fmt)
        s = [slice(None),] * l

        if r  is None: r  = slice(None)
        if p1 is None: p1 = slice(None)
        if p1 is None: p1 = slice(None)
        if w  is None: w  = slice(None)
        if i  is None: i  = slice(None)
        if j  is None: j  = slice(None)

        for I in range(l):
            if fmt[I] == 'r': s[I] = r
            elif fmt[I] == '1': s[I] = p1
            elif fmt[I] == '2': s[I] = p2
            elif fmt[I] == 'w': s[I] = w
            elif fmt[I] == 'i': s[I] = i
            elif fmt[I] == 'j': s[I] = j
            else:
                raise InverterException("Unrecognized format specifier: '{}'.".format(fmt[I]))

        return tuple(s)


    def _setParameters(self, r, p1, p2, w, filename):
        """
        Set the list of Green's function parameters
        """
        # If this is the first Green's function, we just assign
        # all parameters
        if self.r is None:
            self.r  = r
            self.p1 = p1
            self.p2 = p2
            self.w  = w
        else:
            # Else, append the parameter
            if np.any(self.r != r):
                self.r = np.concatenate((self.r, r))
            elif np.any(self.p1 != p1):
                self.p1 = np.concatenate((self.p1, p1))
            elif np.any(self.p2 != p2):
                self.p2 = np.concatenate((self.p2, p2))
            elif np.any(self.w != w):
                self.w = np.concatenate((self.w, w))
            else:
                raise InverterException("All parameters are the same in '{}' as in the first Green's function.".format(filename))
                
        if self.format[self.splitdim] == 'r':
            self.paramlist[filename] = r
        elif self.format[self.splitdim] == '1':
            self.paramlist[filename] = p1
        elif self.format[self.splitdim] == '2':
            self.paramlist[filename] = p2
        elif self.format[self.splitdim] == 'w':
            self.paramlist[filename] = w


