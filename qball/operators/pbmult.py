
from opymize import Variable
from opymize.linear import LinOp

import numba
import numpy as np
from numpy.linalg import norm

try:
    import opymize.tools.gpu
    from opymize.tools.gpu import prepare_kernels
    from pkg_resources import resource_stream
except:
    # no cuda support
    pass

def pbmult_prepare_gpu(P, B, y):
    J = B.shape[0]
    K = y[0]['shape'][1]
    L = B.shape[2]
    M = B.shape[1]
    N = y[0]['shape'][0]
    constvars = {
        'J': J, 'K': K, 'L': L, 'M': M, 'N': N,
        'P': P, 'B': B
    }
    files = [resource_stream('qball.operators', 'pbmult.cu')]
    templates = [
        ("pbmult", "PP", (N, 1, 1), (512, 1, 1)),
        ("bpmult", "PP", (N, J, L), (32, 24, 1))
    ]
    return prepare_kernels(files, templates, constvars)


@numba.njit
def apply_PB(y, P, B, x, precond=False):
    """ Does this: y[P] -= np.einsum('jml,jim->ijl', B, w)
    Unfortunately, advanced indexing without creating a copy is impossible.
    """
    for j in range(x.shape[0]):
        for i in range(x.shape[1]):
            for l in range(B.shape[2]):
                for m in range(x.shape[2]):
                    if precond:
                        y[i,P[j,l]] += np.abs(B[j,m,l])
                    else:
                        y[i,P[j,l]] -= B[j,m,l]*x[j,i,m]

class PBMult(LinOp):
    """ for k,l,i do (Ax)[i,P[j,l]] -= \sum_m B[j,m,l] * x[j,i,m] """
    def __init__(self, K, N, P, B, adjoint=None):
        LinOp.__init__(self)
        self.x = Variable((B.shape[0],N,B.shape[1]))
        self.y = Variable((N,K))
        self.P = P
        self.B = B
        self.B_gpu = None
        if adjoint is None:
            self.adjoint = BPMult(K, N, B, P, adjoint=self)
        else:
            self.adjoint = adjoint
        self._kernel = None

    def prepare_gpu(self, kernels=None):
        if self._kernel is not None: return
        if kernels is None:
            kernels = pbmult_prepare_gpu(self.P, self.B, self.y)
        self._kernel = kernels['pbmult']
        self.adjoint.prepare_gpu(kernels)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernel(x, y)

    def _call_cpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0]
        apply_PB(y, self.P, self.B, x)

    def rowwise_lp(self, y, p=1, add=False):
        assert p is 1
        y = self.y.vars(y)[0]
        x = self.x.vars(self.x.new())[0]
        apply_PB(y, self.P, self.B, x, precond=True)

class BPMult(LinOp):
    """ (Ax)[j,i,l] -= \sum_m B[j,l,m] * x[i,P[j,m]] """
    def __init__(self, K, N, B, P, adjoint=None):
        LinOp.__init__(self)
        self.x = Variable((N,K))
        self.y = Variable((B.shape[0],N,B.shape[1]))
        self.P = P
        self.B = B
        self.B_gpu = None
        if adjoint is None:
            self.adjoint = PBMult(K, N, P, B, adjoint=self)
        else:
            self.adjoint = adjoint
        self._kernel = None

    def prepare_gpu(self, kernels=None):
        if self._kernel is not None: return
        if kernels is None:
            kernels = pbmult_prepare_gpu(self.P, self.B, self.x)
        self._kernel = kernels['bpmult']
        self.adjoint.prepare_gpu(kernels)

    def _call_gpu(self, x, y=None, add=False):
        assert add
        assert y is not None
        self._kernel(x, y)

    def _call_cpu(self, x, y=None, add=False):
        assert add
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0]
        y -= np.einsum('jlm,ijm->jil', self.B, x[:,self.P])

    def rowwise_lp(self, y, p=1, add=False):
        assert p is 1
        assert add
        y = self.y.vars(y)[0]
        y += norm(self.B, ord=1, axis=2)[:,None,:]
