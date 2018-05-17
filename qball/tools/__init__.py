
import numpy as np
import numba

def clip_hardi_data(data, delta=1e-5):
    """ Apply thresholding from Aganj 2010, equation 19. """
    I1 = (data < 0)
    I2 = (0 <= data) & (data < delta)
    I4 = (1-delta <= data) & (data < 1)
    I5 = (1.0 <= data)
    data[I1] = delta/2
    data[I2] = delta/2 + data[I2]**2/(2*delta)
    data[I4] = (1 - delta/2) - (1 - data[I4])**2/(2*delta)
    data[I5] = (1 - delta/2)

def truncate(x, n):
    k = -int(np.floor(np.log10(abs(x))))
    # Example: x = 0.006142 => k = 3 / x = 2341.2 => k = -3
    k += n - 1
    if k > 0:
        x_str = str(abs(x))[:(k+2)]
    else:
        x_str = str(abs(x))[:n]+"0"*(-k)
    return np.sign(x)*float(x_str)

def normalize_odf(odf, vol):
    odf_flat = odf.reshape(odf.shape[0], -1)
    odf_sum = np.einsum('k,ki->i', vol, odf_flat)
    odf_flat[:] = np.einsum('i,ki->ki', 1.0/odf_sum, odf_flat)

@numba.njit
def apply_PB(pgrad, P, B, w, precond=False, inpaint_nloc=np.empty(0, dtype=np.bool)):
    """ Does this: pgrad[P] -= np.einsum('jlm,ijlt->jmti', B, w)
    Unfortunately, advanced indexing without creating a copy is impossible.
    """
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            for l in range(w.shape[2]):
                for m in range(B.shape[2]):
                    for t in range(w.shape[3]):
                        if inpaint_nloc.size == 0 or inpaint_nloc[i]:
                            if precond:
                                pgrad[P[j,m],t,i] += np.abs(B[j,l,m])
                            else:
                                pgrad[P[j,m],t,i] -= B[j,l,m]*w[i,j,l,t]
