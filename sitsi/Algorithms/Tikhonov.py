r"""
This module calculates the best fitting radial profile for a given set of input
data, using Tikhonov regularization. The general least-squares problem we
consider is

.. math::
    
    \mathrm{min} \sum_i\left\lVert I_i^{\rm exp} - I_i \right\rVert_2^2

where :math:`I_i^{\rm exp}` is the experimental data and :math:`I_i=I_i(x)` the
synthetic data resulting from the least-squares solution :math:`x`. In
``sitsi``, we take :math:`I_i` to be

.. math::
    
    I_i = \sum_j G_{ij} x^j,

where :math:`G_{ij}` is a `SOFT2 <https://github.com/hoppe93/SOFT2>`_ Green's
function. We regularize this problem by adding a scaled matrix term
:math:`\alpha\Gamma_{ij}`:

.. math::

    \mathrm{min}\left[ \sum_i\left\lVert I_i^{\rm exp} - \sum_j G_{ij} x^j \right\rVert_2^2 + \left\lVert \sum_j \alpha\Gamma_{ij}x^j \right\rVert_2^2 \right]

The simplest choice for :math:`\Gamma_{ij}` is to use an identity matrix. We
also implement a finite difference operator in ``sitsi``. The scale factor
:math:`alpha` is determined using the L-curve method
(https://www.sintef.no/globalassets/project/evitameeting/2005/lcurve.pdf).

"""

import numpy as np
from .. InverterException import InverterException


class Tikhonov:
    

    def __init__(self, inp, method='standard', fitness=None):
        """
        Constructor.

        method:  Name of Tikhonov method to use. Either 'standard' (uses a
                 constant times an identity matrix for regularization), or
                 'diff' (uses forward finite difference for regularization)
        fitness: Fitness function to use, taking two input arguments:
                   (1) the input data, (2) the best fit output.
                 The default is to take the sum of differences squared, i.e.
                   sum(|a-b|^2)
                 where a and b are the input and output vectors respectively.
        inp:     List of tuples, with each tuple consisting of the input data
                 as well as the Green's function which can be used to
                 generate synthetic data for the input data.
        """
        self.data  = []
        self.green = []
        self.fitness = fitness

        if not self.checkMethod(method.lower()):
            raise InverterException("Unrecognized method specified: '{}'.".format(method))
        self.method = method

        if self.fitness is None:
            self.fitness = lambda inp, synth : np.sum(np.abs(inp - synth)**2)

        # Store input data and Green's functions
        for i in inp:
            self.data.append(i[0])
            self.green.append(i[1])

        self.data = np.concatenate(self.data)
        self.green = np.concatenate(self.green)

        if self.data.size != self.green.shape[1]:
            raise InverterException("Incompatible dimensions of input data and Green's function.")


    def checkMethod(self, method):
        """
        Checks if the specified Tikhonov method is valid.
        """
        return (method in ['diff', 'standard', 'svd'])


    def invert(self):
        """
        Solves for the optimum using a Tikhonov method.
        Returns a tuple consisting of the solution and the solution
        multiplied with the input Green's function.
        """
        invfunc = None
        if self.method == 'diff':
            invfunc = self._invert_general
            self._invert_general_init('diff')
        elif self.method == 'standard':
            invfunc = self._invert_general
            self._invert_general_init('standard')
        elif self.method == 'svd':
            invfunc = self._invert_svd
            self._invert_svd_init()
        else:
            raise InverterException("Unrecognized method specified: '{}'.".format(self.method))

        def evaluate(alpha):
            _, Ax = invfunc(alpha)
            return self.fitness(self.data, Ax)

        lower, upper = -100, 100
        minimum = evaluate(10.0 ** lower)
        maximum = evaluate(10.0 ** upper)

        tol    = 1e-4
        tol_it = 0.1

        def is_good(alpha):
            fitness = evaluate(alpha)
            return ((fitness - minimum) / (maximum-minimum)) < tol

        # L-curve method
        while (upper - lower) > tol_it:
            mid = (upper + lower) / 2
            if is_good(10.0 ** mid):
                lower = mid
            else:
                upper = mid

        x, Ax = invfunc(10.0 ** lower)

        return x, Ax

    
    def _invert_general_init(self, method='standard'):
        """
        Initializes the general Tikhonov methods.
        """
        N = self.green.shape[0]

        # SELECT OPERATOR TO ADD
        if method == 'diff':
            # (Upwind) finite difference
            self.diff_D = (np.eye(N) - np.eye(N, k=1))[:-1]
        elif method == 'standard':
            # Scaled identity matrix
            self.diff_D = np.eye(N)
        else:
            raise InverterException("Unrecognized generalized Tikhonov method specified: '{}'.".format(method))

        # Set up input vector
        self.diff_b = np.hstack((self.data, np.zeros(self.diff_D.shape[0])))


    def _invert_general(self, alpha):
        """
        Solves for the optimum using a Tikhonov method, with a scaled term
        added to the equation. I.e, instead of solving the ill-posed problem

          min || A*x - b ||^2
        
        we solve

          min || A*x - b + alpha*D ||^2
        """
        # Construct matrix to invert
        A = np.vstack((self.green.T, alpha * self.diff_D))

        x, _, _, _ = np.linalg.lstsq(A, self.diff_b, rcond=None)
        img = self.green.T.dot(x)

        return x, img


    def _invert_svd_init(self):
        """
        Initializes the SVD method for Tikhonov regularization.
        """
        self.svd_u, self.svd_s, self.svd_vt = np.linalg.svd(self.green.T, full_matrices=False)


    def _invert_svd(self, alpha):
        """
        Solves the linear problem using Tikhonov regularization and
        SVD decomposition of the linear operator matrix.
        """
        s = np.copy(self.svd_s)
        f = s**2 / (s**2 + alpha**2)
        s = np.divide(1, s, where=(s>0))
        s = s*f

        pinv = np.matmul(self.svd_vt.T, np.multiply(s[...,np.newaxis], self.svd_u.T))

        x   = pinv.dot(self.data)
        img = self.green.T.dot(x)

        return x, img


