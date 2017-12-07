
import logging
import numpy as np

try:
    from skimage.filters import threshold_otsu as otsu
except:
    from dipy.segment.threshold import otsu

from scipy.stats import rice, chi2
from scipy.optimize import brentq, minimize_scalar
from scipy.special import i0

from functools import partial
from multiprocessing import Pool

from qball.tools import matrix2brl

def rice_paramci(d, sigma, alpha=None, thresh=None):
    """ Estimate structural parameter of a Rice distribution

    The estimation uses the likelihood ratio test (LRT) and assumes the scaling
    parameter sigma to be known. It requires only a single sample d. One of
    alpha (confidence level) or thresh are required.

    Args:
        d : sample value
        sigma : (known) scaling parameter of Rice distribution
        alpha : confidence level
        thresh : LRT threshold derived from ppf of chi^2 distribution

    Returns:
        optimal_nu : MLE estimator of nu
        nu1, nu2 : Confidence interval for the estimated nu
    """
    assert(alpha is not None or thresh is not None)
    if thresh is None:
        thresh = np.exp(-0.5*chi2.ppf(1.0-alpha, 1))
    sigma_sq_i = 1.0/(sigma*sigma)
    result = [0,0,0]

    # the following helper function is derived from the (negative) Rice pdf
    #   rice.pdf(x,n,s) = x/s**2 * exp(-(x**2 + n**2)/(2*s**2)) * I[0](x*n/s**2)
    # skipping factors that don't depend on nu
    func_pdf = lambda nu: -np.exp(-0.5*nu*nu*sigma_sq_i)*i0(nu*d*sigma_sq_i)

    # estimate optimal_nu using MLE
    #
    # the built-in method for MLE is comparably slow:
    #   optimal_nu = rice.fit(d, floc=0, fscale=rice_scale)[0]*rice_scale
    #
    # solve explicit minimization problem instead:
    opt_res = minimize_scalar(func_pdf, bounds=(0.0,1.0),
        method='bounded', options={ 'xatol': np.sqrt(np.spacing(1)) })
    assert(opt_res.success)
    optimal_nu = opt_res.x
    assert(optimal_nu >= 0.0 and optimal_nu <= 1.0)

    # determine confidence intervals for optimal_nu using LRT
    #
    # helper function derived from the (shifted) Rice likelihood ratio
    shift = func_pdf(optimal_nu)*thresh
    func_lrt = lambda nu: shift - func_pdf(nu)

    # func_lrt has a (positive) maximum at optimal_nu
    # we're intrested in the interval, where func_lrt is positive
    result[0] = optimal_nu

    # determine root left of optimal_nu:
    if func_lrt(np.spacing(1)) < 0:
        result[1] = brentq(func_lrt, np.spacing(1), optimal_nu)
    else:
        result[1] = np.spacing(1)

    # determine root right of optimal_nu:
    if func_lrt(1.0 - np.spacing(1)) < 0:
        result[2] = brentq(func_lrt, optimal_nu, 1.0 - np.spacing(1))
    else:
        result[2] = 1.0 - np.spacing(1)

    return tuple(result)

def clip_data(data, delta=1e-5):
    """ Apply thresholding from Aganj 2010, equation 19. """
    I1 = (data < 0)
    I2 = (0 <= data) & (data < delta)
    I4 = (1-delta <= data) & (data < 1)
    I5 = (1.0 < data)
    data[I1] = delta/2
    data[I2] = delta/2 + data[I2]**2/(2*delta)
    data[I4] = 1 - delta/2 - (1 - data[I4])**2/(2*delta)
    data[I5] = 1 - delta/2

def compute_bounds(b_sph, data, alpha, mask=None):
    """ Compute fidelity bounds for HARDI signal `data`.

    Args:
        b_sph : Sphere object from b-vectors
        data : HARDI signal
        alpha : confidence level

    Returns:
        fl, fu : lower and upper bound for averaged log(-log(data))
    """
    imagedims = data.shape[:-1]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    l_labels = b_sph.mdims['l_labels']
    assert(data.shape[-1] == l_labels)

    if mask is None:
        # automatically estimate foreground from histogram thresholding (Otsu)
        mask = np.mean(data, axis=-1)
        thresh = otsu(mask)
        mask = (mask <= thresh)
        print(matrix2brl(mask.astype(int)))

    n_samples = np.sum(np.logical_not(mask))
    samples = data[np.logical_not(mask.reshape(imagedims))]
    assert(samples.shape == (n_samples,l_labels))

    logging.debug('Computing confidence intervals with confidence level %.3f ...', alpha)

    # even though we know `rice_nu == 0.406569659741`, we cannot fix this in
    # the parameter estimation provided by SciPy
    rice_nu, _, rice_scale = rice.fit(samples[:], floc=0)

    logging.debug('Estimated sigma=%.5f from n=%d samples.', rice_scale, n_samples)

    # Compute confidence intervals using the likelihood ratio test (LRT)
    data_l = np.zeros(data.size)
    data_u = np.zeros(data.size)
    thresh = np.exp(-0.5*chi2.ppf(1.0-alpha, 1))
    paramci_partial = partial(rice_paramci, sigma=rice_scale, thresh=thresh)

    # parallelize using all available CPU cores
    p = Pool(processes=None)
    res = p.map(paramci_partial, data.ravel())
    p.terminate()
    data_nu, data_l[:], data_u[:] = np.array(res).T

    clip_data(data_l)
    clip_data(data_u)
    assert((data_l <= data_u).all())

    fl = np.zeros((l_labels, n_image), order='C')
    fu = np.zeros((l_labels, n_image), order='C')

    fl[:] = np.log(-np.log(data_u)).reshape(-1, l_labels).T
    fu[:] = np.log(-np.log(data_l)).reshape(-1, l_labels).T
    assert((fl <= fu).all())

    fl_mean = np.einsum('ki,k->i', fl, b_sph.b)/(4*np.pi)
    fu_mean = np.einsum('ki,k->i', fu, b_sph.b)/(4*np.pi)

    fu -= fl_mean
    fl -= fu_mean

    return fl, fu
