"""
This class represents the model

  f(r,p,xi) = f_r(r) * delta(p-p*) * exp(C*xi)

where f_r(r) is an unknown radial density profile, p* is the free momentum
parameter of this model, and C is the free pitch parameter of this model
(p and xi are the momentum and cosine of the pitch angle respectively).

When evaluated with a combination (p*, C), this model returns a new Green's
function which has been multiplied appropriately with a momentum-pitch
distribution function.
"""

import numpy as np
from .. InverterException import InverterException


class DeltaPExpPitch:
    

    def __init__(self, green):
        """
        Constructor.

        green: Green's function to use for evaluating this model.
        """
        self.green = green

        self.pitchidx, self.xi = self._findPitchDimension()


    def eval(self, p, C):
        """
        Evaluates this model with the given parameters p and C
        (which must be scalars).
        """
        if np.asarray(p).size != 1 or np.asarray(C).size != 1:
            raise InverterException("p and C must be scalars.")
    
        pi = self.green.getParameterIndex(p, '1')

        # Delta in p (just get the particular slice of the Green's function)
        gf = self.green.get(p1=pi)

        # Evaluate just distribution function
        f = np.exp(C*self.xi) / np.exp(C) * C

        # Multiply with exponential function (pitch distribution)
        s = [slice(None),] * gf.ndim
        G = 0
        for i in range(len(self.xi)):
            s[self.pitchidx] = i
            G += gf[tuple(s)] * f[i]

        # Remove p dimension
        pdim = self.green.format.find('1')
        if pdim > self.green.format.find('2'):
            pdim -= 1

        s = [slice(None),] * G.ndim
        s[pdim] = 0
        G = G[tuple(s)]

        return G


    def _findPitchDimension(self):
        """
        This method returns the index of the pitch (either xi or thetap)
        dimension in the Green's function.
        """
        pitchname = ['thetap', 'xi']

        idx = -1
        pidx = -1
        name = ""
        data = None
        if self.green.param2name in pitchname:
            name = self.green.param2name
            data = self.green.p2
            idx = self.green.format.find('2')
            pidx = self.green.format.find('1')
        elif self.green.param1name in pitchname:
            name = self.green.param1name
            data = self.green.p1
            idx = self.green.format.find('1')
            pidx = self.green.format.find('2')
        else:
            raise InverterException("The given Green's function does not depend on neither 'thetap' nor 'xi'.")

        # If this is a pitch angle, calculate its cosine
        if name != 'xi':
            data = np.cos(data)

        # If 'p' is "ahead" of the pitch in the Green's function format,
        # then the pitch dimension index will be one less when the p
        # dimension is removed from the Green's function (we do this in
        # 'eval()', since we want a delta function in p)
        #if pidx < idx:
        #    idx -= 1

        return idx, data


