
from opymize import Variable, Functional, Operator

import numpy as np

try:
    import opymize.tools.gpu
    from pycuda import gpuarray
    from pycuda.elementwise import ElementwiseKernel
except:
    # no cuda support
    pass

class BndSSD(Functional):
    """ 0.5*|max(0, f1 - x)|^2 + 0.5*|max(0, x - f2)|^2 """
    def __init__(self, f1, f2, vol=None, mask=None, conj=None):
        Functional.__init__(self)
        self.x = Variable(f1.shape)
        self.f1 = f1
        self.f2 = f2
        self.vol = np.ones(f1.shape[1]) if vol is None else vol
        self.mask = np.ones(f1.shape[0], dtype=bool) if mask is None else mask
        if conj is None:
            cj_vol = 1.0/self.vol
            self.conj = BndSSDConj(f1, f2, vol=cj_vol, mask=mask, conj=self)
        else:
            self.conj = conj
        f1_msk, f2_msk = [np.zeros_like(a) for a in [f1,f2]]
        f1_msk[self.mask,:] = self.vol[None,:]*f1[self.mask,:]
        f2_msk[self.mask,:] = self.vol[None,:]*f2[self.mask,:]
        self._prox = ProxBndSSD(f1_msk, f2_msk)

    def __call__(self, x, grad=False):
        assert not grad
        x = self.x.vars(x)[0]
        posval1 = np.fmax(0.0, (self.f1 - x)[self.mask,:])
        posval2 = np.fmax(0.0, (x - self.f2)[self.mask,:])
        val = 0.5*np.einsum('ik,k->', posval1**2, self.vol) \
            + 0.5*np.einsum('ik,k->', posval2**2, self.vol)
        infeas = 0
        return (val, infeas)

    def prox(self, tau):
        msk = self.mask
        tauvol = np.zeros_like(self.f1)
        tauvol[msk,:] = (tau*np.ones(self.f1.size)).reshape(self.f1.shape)[msk,:]
        tauvol[msk,:] = np.einsum('ik,k->ik', tauvol[msk,:], self.vol)
        tauvol += 1.0
        self._prox.alpha = tauvol
        self._prox.tau = tau
        return self._prox

class BndSSDConj(Functional):
    """ 0.5*<x,x>_b + sum[max(f1*x,f2*x)]
        same as 0.5*<x,x>_b + sum[(a + sign(x)*b)*x]
        where a = (f1+f2)/2 and b = (f2-f1)/2
    """
    def __init__(self, f1, f2, vol=None, mask=None, conj=None):
        Functional.__init__(self)
        self.x = Variable(f1.shape)
        self.a = 0.5*(f1 + f2)
        self.b = 0.5*(f2 - f1)
        self.vol = np.ones(f1.shape[1]) if vol is None else vol
        self.mask = np.ones(f1.shape[0], dtype=bool) if mask is None else mask
        if conj is None:
            cj_vol = 1.0/self.vol
            self.conj = BndSSD(f1, f2, vol=cj_vol, mask=mask, conj=self)
        else:
            self.conj = conj

    def __call__(self, x, grad=False):
        assert not grad
        x = self.x.vars(x)[0]
        x_msk = x[self.mask,:]
        a_msk, b_msk = self.a[self.mask,:], self.b[self.mask,:]
        val = 0.5*np.einsum('ik,k->', x_msk**2, self.vol) \
            + np.einsum('ik,ik->', a_msk + np.sign(x_msk)*b_msk, x_msk)
        infeas = 0
        return (val, infeas)

class ProxBndSSD(Operator):
    """ y = max(x + tau*f1, min(x + tau*f2, alpha*x))/alpha """
    def __init__(self, f1, f2, alpha=2.0, tau=1.0):
        Operator.__init__(self)
        self.x = Variable(f1.shape)
        self.y = Variable(f1.shape)
        self.f1 = f1
        self.f2 = f2
        self.alpha = alpha
        self.tau = tau

    def prepare_gpu(self, type_t="double"):
        np_dtype = np.float64 if type_t == "double" else np.float32
        taufact = "tau[i]*" if type(self.tau) is np.ndarray else "tau[0]*"
        self.gpuvars = {
            'f1':       gpuarray.to_gpu(self.f1),
            'f2':       gpuarray.to_gpu(self.f2),
            'alpha':    gpuarray.to_gpu(self.alpha),
            'tau':      gpuarray.to_gpu(np.asarray(self.tau, dtype=np_dtype))
        }
        headstr  = "{0} *x, {0} *y, "
        headstr += "{0} *f1, {0} *f2, {0} *alpha, {0} *tau"
        self._kernel = ElementwiseKernel(headstr.format(type_t),
            ("y[i] = fmax(x[i] + {0}f1[i], "\
                    + "fmin(x[i] + {0}f2[i], "\
                        + "alpha[i]*x[i]))/alpha[i]").format(taufact))

    def _call_gpu(self, x, y=None, add=False, jacobian=False):
        assert not jacobian
        assert not add
        g = self.gpuvars
        y = x if y is None else y
        self._kernel(x, y, g['f1'], g['f2'], g['alpha'], g['tau'])

    def _call_cpu(self, x, y=None, add=False, jacobian=False):
        assert not jacobian
        assert not add
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0] if y is not None else x
        np.fmax(x + self.tau*self.f1,
                np.fmin(x + self.tau*self.f2, self.alpha*x), out=y)
        y /= self.alpha
