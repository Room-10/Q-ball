
"""
Set of functions to compute the gradients Duk of u^k (or its adjoint operator)
on a staggered grid and setting `p^kt += b^k * Duk^t` (or
`u^k -= b^k * div(p^k)`). The "boundary" of p is left untouched.

Example in two dimensions (x are the grid points of u), k fixed:

       |                |
... -- x -- -- v1 -- -- x -- ...
       |                |          v1 and w1 -- the partial x-derivatives at top and bottom
       |                |          v2 and w2 -- the partial y-derivatives at left and right
       v2      Du       w2         Du -- the gradient at the center of the box
       |                |          (Du)_1 -- mean value of v1 and w1
       |                |          (Du)_2 -- mean value of v2 and w2
... -- x -- -- w1 -- -- x -- ...
       |                |

In one dimension there is no averaging, in three dimensions each
derivative is the mean value of four finite differences etc.

The adjoint operator (reading from p and writing to u) is understood to be the
negative divergence -div(p^k) with Dirichlet boundary.
"""

import numpy as np
import itertools

from numba import jit

def staggered_diff_avgskips(imagedims):
    d_image = len(imagedims)
    skips = tuple(int(np.prod(imagedims[(d_image-t):])) for t in range(d_image))
    navgskips =  1 << (d_image - 1)
    avgskips = np.zeros([d_image, navgskips], dtype=np.int64, order='C')
    for t in range(d_image):
        for m, p in enumerate(itertools.product([0, 1], repeat=(d_image - 1))):
            avgskips[t,m] = np.inner(p[:t] + (0,) + p[t:], skips)
    return avgskips

def gradient(pgrad, u, b, avgskips):
    staggered_diff(pgrad, u, b, avgskips, adjoint=False)

def divergence(p, ugrad, b, avgskips):
    staggered_diff(p, ugrad, b, avgskips, adjoint=True)

@jit
def staggered_diff(p, u, b, avgskips, adjoint=False):
    """
    Computes `p^kt += b^k * Duk^t`.

    Args:
        p : array where the result is written as `p^kt += b^k * Duk^t`.
        u : function to be derived
        b : factor such that `p^kt += b^k * Duk^t`
        avgskips : output of staggered_diff_avgskips(u.shape[1:])
        adjoint : (optional) if True, apply the adjoint operator (reading from p
                  and writing to u), i.e. `u^k -= b^k * div(p^k)`.
    """
    l_labels, d_image, n_image = p.shape
    navgskips =  1 << (d_image - 1)
    imagedims = u.shape[1:]

    skips = (1,)
    for t in range(1,d_image):
        skips += (skips[-1]*imagedims[d_image-t],)

    u_flat = u.reshape(l_labels, n_image)
    coords = np.zeros(d_image, dtype=np.int64)

    for k in range(l_labels):
        for t in range(d_image):
            coords *= 0
            for i in range(n_image):
                # ignore boundary points
                in_range = True
                for dc in reversed(range(d_image)):
                    if coords[dc] >= imagedims[dc] - 1:
                        in_range = False
                        break

                if in_range:
                    # regular case
                    pk = p[k]
                    uk = u_flat[k]
                    bk = b[k]/navgskips

                    for avgskip in avgskips[t]:
                        base = i + avgskip
                        if adjoint:
                            uk[base + skips[t]] += bk * pk[t,i]
                            uk[base] -= bk * pk[t,i]
                        else:
                            pk[t,i] += bk * uk[base + skips[t]]
                            pk[t,i] -= bk * uk[base]

                # advance coordinates
                for dd in reversed(range(d_image)):
                    coords[dd] += 1
                    if coords[dd] >= imagedims[dd]:
                        coords[dd] = 0
                    else:
                        break
