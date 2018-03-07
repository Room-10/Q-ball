
import numpy as np
from numpy.linalg import norm

def block_normest(x, y, op, opadj, tol=1.0e-4, maxits=500):
    """ Estimate the spectral norm of the given linear operator op/opadj.
    """
    m, n = y.data.size, x.data.size
    itn = 0

    # Compute an estimate of the abs-val column sums.
    y[:] = np.ones(m)
    y[np.random.randn(m) < 0] = -1
    opadj(x,y)
    x[:] = abs(x.data)

    # Normalize the starting vector.
    e = norm(x.data)
    if e < 1e-13:
        return e, itn
    x *= 1.0/e
    e0 = 0
    while abs(e-e0) > tol*e:
        e0 = e
        op(x, y)
        normy = norm(y.data)
        if normy < 1e-13:
            y[:] = np.random.rand(m)
            normy = norm(y.data)
        opadj(x, y)
        normx = norm(x.data)
        e = normx/normy
        x *= 1.0/normx
        itn += 1
        if itn > maxits:
            print("Warning: normest didn't converge!")
            break
    return e, itn

class BlockVar(object):
    def __init__(self, *args):
        self._descr = {}
        self._args = args
        for a in args:
            self._append(*a)
        self.size = self._size() # size is fixed after init
        self.data = np.zeros((self.size,), order='C')

    def _append(self, name, dim):
        offset = self._size()
        self._descr[name] = (offset, dim)

    def _size(self):
        return sum([np.prod(d[1]) for d in self._descr.values()])

    def copy(self):
        cpy = BlockVar(*self._args)
        cpy.data[:] = self.data
        return cpy

    def vars(self):
        return [self[a[0]] for a in self._args]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            offset, dim = self._descr[idx]
            size = np.prod(dim)
            return self.data[offset:offset+size].reshape(dim)
        else:
            return self.data[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, str):
            offset, dim = self._descr[idx]
            size = np.prod(dim)
            self.data[offset:offset+size] = value
        else:
            if isinstance(value, BlockVar):
                value = value.data
            self.data[idx] = value

    def __sub__(self, other):
        return self + (-1.0)*other

    def __add__(self, other):
        if isinstance(other, BlockVar):
            other = other.data
        out = self.copy()
        out.data[:] += other
        return out

    def __rmul__(self, scalar):
        out = self.copy()
        out.data[:] *= scalar
        return out

    def __mul__(self, scalar):
        return scalar*self

    def __str__(self):
        return self.data.__str__()

    def __iter__(self):
        for d in self._args:
            yield { 'name': d[0], 'offset': self._descr[d[0]][0] }
