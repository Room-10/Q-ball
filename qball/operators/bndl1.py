
from opymize import Variable, Functional, Operator

import numpy as np
from numpy.linalg import norm

try:
    import opymize.tools.gpu
    from pycuda import gpuarray
    from pycuda.elementwise import ElementwiseKernel
except:
    # no cuda support
    pass

class MaxFct(Functional):
    """ \sum_ik b[k]*max(0, x[i,k] - f[i,k]) """
    def __init__(self, data, vol=None, mask=None, conj=None):
        Functional.__init__(self)
        self.f = np.atleast_2d(data)
        self.x = Variable(self.f.shape)
        self.vol = np.ones(data.shape[1]) if vol is None else vol
        self.mask = np.ones(data.shape[0], dtype=bool) if mask is None else mask
        if conj is None:
            self.conj = MaxFctConj(data, weights=vol, mask=mask, conj=self)
        else:
            self.conj = conj

    def __call__(self, x, grad=False):
        x = self.x.vars(x)[0]
        posval = np.fmax(0, (x - self.f)[self.mask,:])
        val = np.einsum('ik,k->', posval, self.vol)
        infeas = 0
        result = (val, infeas)
        if grad:
            df = np.zeros_like(x)
            posval[posval > 0] = 1.0
            df[self.mask,:] = np.einsum('ik,k->', posval, self.vol)
            return result, df
        else:
            return result

class MaxFctConj(Functional):
    """ sum_i <x[i,:],f[i,:]> + \delta_{0 <= b[k]*x[i,k] <= 1} """
    def __init__(self, data, weights=None, mask=None, conj=None):
        Functional.__init__(self)
        self.f = np.atleast_2d(data)
        self.x = Variable(self.f.shape)
        self.weights = np.ones(data.shape[1]) if weights is None else weights
        self.mask = np.ones(data.shape[0], dtype=bool) if mask is None else mask
        if conj is None:
            self.conj = MaxFct(data, vol=vol, mask=mask, conj=self)
        else:
            self.conj = conj
        self._prox = IntervProj(self.weights, self.f, mask=mask)

    def __call__(self, x, grad=False):
        x = self.x.vars(x)[0]
        val = np.einsum('ik,ik->', x[self.mask,:], self.f[self.mask,:])
        infeas = norm(np.fmin(0, x[self.mask,:]), ord=np.inf)
        bx = np.einsum('ik,k->ik', x[self.mask,:], self.weights)
        infeas += norm(np.fmin(0, 1.0 - bx), ord=np.inf)
        result = (val, infeas)
        if grad:
            df = np.zeros_like(x)
            df[self.mask,:]= self.f[self.mask,:]
            result = result, df
        return result

    def prox(self, tau):
        self._prox.a = tau
        return self._prox

class IntervProj(Operator):
    """ y[i,k] = proj_[0,b[k]](x[i,k] - a[i,k]*shift[i,k]) """
    def __init__(self, b, shift, a=1, mask=None):
        Operator.__init__(self)
        assert b.size == shift.shape[1]
        self.x = Variable(shift.shape)
        self.y = Variable(shift.shape)
        self.b = b
        self.a = a
        self.shift = shift
        self.mask = np.ones(shift.shape[0], dtype=bool) if mask is None else mask

    def prepare_gpu(self, type_t="double"):
        # don't multiply with a if a is 1 (not 1.0!)
        afact = "" if self.a is 1 else "a[0]*"
        if type(self.a) is np.ndarray:
            afact = "a[i]*"

        np_dtype = np.float64 if type_t == "double" else np.float32
        self.gpuvars = {
            'b':        gpuarray.to_gpu(self.b),
            'shift':    gpuarray.to_gpu(self.shift),
            'a':        gpuarray.to_gpu(np.asarray(self.a, dtype=np_dtype))
        }
        headstr = "{0} *x, {0} *y, {0} *shift, {0} *a, {0} *b".format(type_t)
        self._kernel = ElementwiseKernel(headstr,
            "y[i] = fmax(0.0, fmin(b[i%{}], x[i] - {}shift[i]))"\
                .format(self.b.size, afact))

    def _call_gpu(self, x, y=None, add=False, jacobian=False):
        assert not jacobian
        assert not add
        g = self.gpuvars
        y = x if y is None else y
        self._kernel(x, y, g['shift'], g['a'], g['b'])

    def _call_cpu(self, x, y=None, add=False, jacobian=False):
        assert not jacobian
        assert not add
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0] if y is not None else x
        np.fmax(0.0, np.fmin(self.b[None,:], x - self.a*self.shift), out=y)

