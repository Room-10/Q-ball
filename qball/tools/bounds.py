
import logging
import numpy as np
from dipy.segment.mask import median_otsu

def compute_bounds(b_sph, data, c=0.7):
    """ Compute fidelity bounds for HARDI signal `data`.

    Args:
        b_sph : Sphere object from b-vectors
        data : HARDI signal
        c : (optional) anticipated fraction of statistical outliers

    Returns:
        fl, fu : lower and upper bound for averaged log(-log(data))
    """
    imagedims = data.shape[:-1]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    l_labels = b_sph.mdims['l_labels']
    assert(data.shape[-1] == l_labels)

#    three_d_data = data[(slice(None),)*d_image + (None,)*(3-d_image) + (slice(None),)]
#    maskdata, mask = median_otsu(three_d_data,dilate=3)
#    logging.info('Foreground mask')
#    logging.info(mask.astype(int))

    mask = np.rot90(np.array([
        [ 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        [ 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
        [ 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
        [ 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
        [ 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [ 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [ 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [ 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    ], dtype=bool), k=3)

    n_samples = np.sum(np.logical_not(mask))

    samples = data[np.logical_not(mask.reshape(imagedims))]
    samples -= 0.406569659741
    assert(samples.shape == (n_samples,l_labels))

    noise_l = np.percentile(samples, 100*c/2, axis=0)
    noise_u = np.percentile(samples, 100*(1.0-c/2), axis=0)

    logging.debug('Bounds: n_samples = %d, noise in (%.5f, %.5f)', \
        n_samples, noise_l.min(), noise_u.max())

    data_l_clipped = np.clip(data - noise_u, np.spacing(1), 1-np.spacing(1))
    data_u_clipped = np.clip(data - noise_l, np.spacing(1), 1-np.spacing(1))
    assert((data_l_clipped <= data_u_clipped).all())

    fl = np.zeros((l_labels, n_image), order='C')
    fu = np.zeros((l_labels, n_image), order='C')

    fl[:] = np.log(-np.log(data_u_clipped)).reshape(-1, l_labels).T
    fu[:] = np.log(-np.log(data_l_clipped)).reshape(-1, l_labels).T
    assert((fl <= fu).all())

    fl_mean = np.einsum('ki,k->i', fl, b_sph.b)/(4*np.pi)
    fu_mean = np.einsum('ki,k->i', fu, b_sph.b)/(4*np.pi)

    fu -= fl_mean
    fl -= fu_mean

    return fl, fu