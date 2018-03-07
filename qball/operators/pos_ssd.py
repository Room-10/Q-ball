
from opymize import Variable, Functional, Operator

import numpy as np

try:
    import opymize.tools.gpu
    from pycuda import gpuarray
    from pycuda.elementwise import ElementwiseKernel
except:
    # no cuda support
    pass

class PosSSD(Functional):
    """ 0.5*<x-f, x-f>_b + shift + \delta_{u >= 0}
        where b is the volume element
    """
    def __init__(self, data, vol=None, mask=None, conj=None):
        Functional.__init__(self)
        self.x = Variable(data.shape)
        self.f = np.atleast_2d(data)
        self.vol = np.ones(data.shape[1]) if vol is None else vol
        self.mask = np.ones(data.shape[0], dtype=bool) if mask is None else mask
        if conj is None:
            cj_vol = 1.0/self.vol
            cj_data = np.zeros_like(self.f)
            cj_data[self.mask,:] = np.einsum('ik,k->ik', self.f[self.mask,:], self.vol)
            self.conj = PosSSDConj(cj_data, vol=cj_vol, mask=mask, conj=self)
        else:
            self.conj = conj
        prox_shift = np.zeros_like(self.f)
        prox_shift[self.mask,:] = self.f[self.mask,:]
        prox_shift = prox_shift.ravel()
        self._prox = PosShiftScaleOp(self.x.size, prox_shift, 0.5, 1.0)

    def __call__(self, x, grad=False):
        x = self.x.vars(x)[0]
        val = 0.5*np.einsum('ik,k->', (x - self.f)[self.mask,:]**2, self.vol)
        infeas = 0
        result = (val, infeas)
        if grad:
            df = np.zeros_like(x)
            df[self.mask,:] = np.einsum('ik,k->ik', (x - self.f)[self.mask,:], self.vol)
            return result, df.ravel()
        else:
            return result

    def prox(self, tau):
        msk = self.mask
        tauvol = np.zeros_like(self.f)
        tauvol[msk,:] = (tau*np.ones(self.f.size)).reshape(self.f.shape)[msk,:]
        tauvol[msk,:] = np.einsum('ik,k->ik', tauvol[msk,:], self.vol)
        self._prox.a = 1.0/(1.0 + tauvol.ravel())
        self._prox.b = tauvol.ravel()
        return self._prox

class PosSSDConj(Functional):
    """ 0.5*|max(0, x + f)|^2 - 0.5*|f|^2 """
    def __init__(self, data, vol=None, mask=None, conj=None):
        Functional.__init__(self)
        self.x = Variable(data.shape)
        self.f = np.atleast_2d(data)
        self.shift = -0.5*np.einsum('ik,k->', data**2, vol)
        self.vol = np.ones(data.shape[1]) if vol is None else vol
        self.mask = np.ones(data.shape[0], dtype=bool) if mask is None else mask
        if conj is None:
            cj_vol = 1.0/self.vol
            cj_data = np.zeros_like(self.f)
            cj_data[self.mask,:] = np.einsum('ik,k->ik', self.f[self.mask,:], self.vol)
            self.conj = PosSSD(cj_data, vol=cj_vol, mask=mask, conj=self)
        else:
            self.conj = conj

    def __call__(self, x, grad=False):
        assert not grad
        x = self.x.vars(x)[0]
        val = 0.5*np.einsum('ik,k->', np.fmax(0.0, (x + self.f)[self.mask,:])**2, self.vol)
        val += self.shift
        infeas = 0
        return (val, infeas)

class PosShiftScaleOp(Operator):
    """ T(x) = max(0, a*(x + b*shift)), where a, b, shift can be float or arrays """
    def __init__(self, N, shift, a, b):
        Operator.__init__(self)
        self.shift = shift
        self.a = a
        self.b = b
        self.x = Variable(N)
        self.y = Variable(N)

    def prepare_gpu(self):
        # don't multiply with a if a is 1 (not 1.0!)
        afact = "" if self.a is 1 else "a[0]*"
        if type(self.a) is np.ndarray:
            afact = "a[i]*"

        # don't multiply with b if b is 1 (not 1.0!)
        bfact = "" if self.b is 1 else "b[0]*"
        if type(self.b) is np.ndarray:
            bfact = "b[i]*"

        # don't shift if shift is 0 (not 0.0!)
        shiftstr = " + %sshift" % bfact
        shiftstr += "[i]" if type(self.shift) is np.ndarray else "[0]"
        if self.shift is 0 or self.b is 0:
            shiftstr = ""

        self.gpuvars = {
            'shift':    gpuarray.to_gpu(np.asarray(self.shift, dtype=np.float64)),
            'a':        gpuarray.to_gpu(np.asarray(self.a, dtype=np.float64)),
            'b':        gpuarray.to_gpu(np.asarray(self.b, dtype=np.float64))
        }
        headstr = "double *x, double *y, double *shift, double *a, double *b"
        self._kernel = ElementwiseKernel(headstr,
            "y[i] = fmax(0.0, %s(x[i]%s))" % (afact, shiftstr))
        self._kernel_add = ElementwiseKernel(headstr,
            "y[i] += fmax(0.0, %s(x[i]%s))" % (afact, shiftstr))

    def _call_gpu(self, x, y=None, add=False, jacobian=False):
        assert not jacobian
        g = self.gpuvars
        y = x if y is None else y
        krnl = self._kernel_add if add else self._kernel
        krnl(x, y, g['shift'], g['a'], g['b'])

    def _call_cpu(self, x, y=None, add=False, jacobian=False):
        assert not jacobian
        y = x if y is None else y
        if add:
            y += np.fmax(0.0, self.a*(x + self.b*self.shift))
        else:
            y[:] = np.fmax(0.0, self.a*(x + self.b*self.shift))
