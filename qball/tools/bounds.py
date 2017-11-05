
import logging
import numpy as np
from dipy.segment.mask import median_otsu

from scipy.stats import rice, chi2
from scipy.optimize import brentq
from scipy.special import iv

def compute_bounds(b_sph, data, alpha=0.05):
    """ Compute fidelity bounds for HARDI signal `data`.

    Args:
        b_sph : Sphere object from b-vectors
        data : HARDI signal
        alpha : (optional) confidence level, defaults to 0.05

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

    # the mask is rotated by 270 degree (due to default plot being rotated)
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
    assert(samples.shape == (n_samples,l_labels))

    # even though we know `rice_nu == 0.406569659741`, we cannot fix this in
    # the parameter estimation provided by SciPy
    
    # YK: 0.406569659741 is the mean of the background, which is not the same as nu
    rice_nu, _, rice_scale = rice.fit(samples[:], floc=0)

    logging.debug('Bounds: n_samples = %d, rice_sigma = %.5f', n_samples, rice_scale)

    #
    # Compute confidence intervals using the likelihood ratio test (LRT)
    #
    # This is extremely slow (~7 min for ~35000 data points) since it solves
    # two root-finding problems for each of the data points (without any
    # vectorization etc.)
    #
    data_l = np.zeros(data.size)
    data_u = np.zeros(data.size)
    # For LRT, the 2*log of the likelihood ratio is assumed to be chi^2 distributed
    thresh = chi2.ppf(1.0-alpha, 1)
    for i,d in enumerate(data.ravel()):
        # first, estimate nu using MLE
        optimal_nu = rice.fit(d, floc=0, fscale=rice_scale)[0]*rice_scale

        # helper function func is the (shifted) 2*log of the likelihood ratio
        ll_func = lambda nu: np.log(rice.pdf(d,nu/rice_scale,scale=rice_scale))
        optimal_ll = ll_func(optimal_nu)
        func = lambda nu: thresh - 2*(optimal_ll - ll_func(nu))
        
        ll_func1 = lambda nu: iv(nu*d/rice_scale^2)*exp(-nu^2/2/rice_scale^2)
        thresh1 = func1(optimal_nu)*exp(-thresh/2)
        func = lambda nu: thresh1 - ll_func1(nu)
        
        # func has a (positive) maximum at optimal_nu
        # we're intrested in the interval, where func is positive
        # determine root left of optimal_nu:
        if func(np.spacing(1)) < 0:
            data_l[i] = brentq(func, np.spacing(1), optimal_nu)
        else:
            data_l[i] = np.spacing(1)

        # determine root right of optimal_nu:
        if func(1.0 - np.spacing(1)) < 0:
            data_u[i] = brentq(func, optimal_nu, 1.0 - np.spacing(1))
        else:
            data_u[i] = 1.0 - np.spacing(1)

    data_l = data_l.reshape(data.shape)
    data_u = data_u.reshape(data.shape)

    data_l_clipped = np.clip(data_l, np.spacing(1), 1-np.spacing(1))
    data_u_clipped = np.clip(data_u, np.spacing(1), 1-np.spacing(1))
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