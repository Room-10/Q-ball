
import numpy as np

def compute_bounds(b_sph, data, c=0.05):
    imagedims = data.shape[:-1]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    l_labels = b_sph.mdims['l_labels']
    assert(data.shape[-1] == l_labels)

    f = np.zeros((l_labels, n_image), order='C')
    fl = np.zeros((l_labels, n_image), order='C')
    fu = np.zeros((l_labels, n_image), order='C')

    data_clipped = np.clip(data, np.spacing(1), 1-np.spacing(1))
    loglog_data = np.log(-np.log(data_clipped))
    f[:] = loglog_data.reshape(-1, l_labels).T

    for i in range(n_image):
        for k in range(l_labels):
            if f[k,i] > 0:
                fl[k,i] = f[k,i]*(1.0-c)
                fu[k,i] = f[k,i]*(1.0+c)
            else:
                fl[k,i] = f[k,i]*(1.0+c)
                fu[k,i] = f[k,i]*(1.0-c)

    assert((fl <= fu).all())

    fl_mean = np.einsum('ki,k->i', fl, b_sph.b)/(4*np.pi)
    fu_mean = np.einsum('ki,k->i', fu, b_sph.b)/(4*np.pi)

    fu -= fl_mean
    fl -= fu_mean

    return fl, fu