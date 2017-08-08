
import numpy as np

class BlockVar(object):
    def __init__(self, *args):
        self._descr = {}
        self._args = args
        for a in args:
            self._append(*a)
        self.data = np.zeros((self._size(),), order='C')

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
        if isinstance(idx, slice):
            return self.data[idx]
        else:
            offset, dim = self._descr[idx]
            size = np.prod(dim)
            return self.data[offset:offset+size].reshape(dim)

    def __setitem__(self, idx, value):
        if isinstance(idx, slice):
            if isinstance(value, BlockVar):
                value = value.data
            self.data[idx] = value
        else:
            offset, dim = self._descr[idx]
            size = np.prod(dim)
            self.data[offset:offset+size] = value

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
