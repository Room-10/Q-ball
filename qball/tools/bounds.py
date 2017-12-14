
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

from qball.tools import matrix2brl, clip_hardi_data

def logi0_large(x):
    return x - 0.5*np.log(2*np.pi*x)

def logi0_appx(x, thresh=700, pad=50):
    """ Approximation of logarithm of modified Bessel function of order 0

    Returns `log(I[0](x))`, exactly for `x + pad < thresh` and approximately for
    `x > thresh`, guaranteeing for a smooth transition in between.

    `I[0](x)` is approximated by `exp(x)/sqrt(2*PI*x)` for `x > thresh`.
    """
    x = np.array(x)
    result = np.zeros_like(x)

    I = (x <= thresh - pad)
    result[I] = np.log(i0(x[I]))

    I = (x >= thresh)
    result[I] = logi0_large(x[I])

    I = (x > thresh - pad) & (x < thresh)
    lbd = (thresh - x[I])/pad
    result[I] = (1 - lbd)*logi0_large(x[I]) + lbd*np.log(i0(x[I]))

    return result

def rice_nu_paramci(d, sigma, alpha=None, thresh=None):
    """ Estimate structural parameter `nu` of a Rice distribution

    The estimation uses the likelihood ratio test (LRT) and assumes the scaling
    parameter sigma to be known. It requires only a single sample `d`, but
    accuracy improves for more sample values. One of
    `alpha` (confidence level) or `thresh` are required.

    Args:
        d : one-dim. numpy array, sample values
        sigma : (known) scaling parameter of Rice distribution
        alpha : confidence level
        thresh : LRT threshold derived from ppf of chi^2 distribution

    Returns:
        optimal_nu : MLE estimator of nu
        nu1, nu2 : Confidence interval for the estimated nu
    """
    assert(len(d.shape) == 1)
    assert(alpha is not None or thresh is not None)
    if thresh is None:
        thresh = chi2.ppf(1.0-alpha, 1)
    sigma_sq_i = 1.0/(sigma*sigma)
    result = [0,0,0]

    # the following helper function is derived from the (negative) Rice pdf
    #   rice.pdf(x,n,s) = x/s**2 * exp(-(x**2 + n**2)/(2*s**2)) * I[0](x*n/s**2)
    # skipping factors that don't depend on nu
    func_pdf = lambda nu: -np.sum(
        -0.5*nu*nu*sigma_sq_i + logi0_appx(nu*d*sigma_sq_i)
    )

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
    shift = func_pdf(optimal_nu) + 0.5*thresh
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

def rice_nu_paramci_batch(data, sigma, alpha):
    """ Compute `alpha` fidelity bounds for `data` corrupted with Rician noise

    Args:
        data : numpy array of shape (B,N)
            N values, B iid noisy samples each
        sigma : scaling parameter of underlying Rice distribution
        alpha : confidence level

    Returns:
        data_nu : maximum-likelihood estimate
        data_l, data_u : numpy arrays of shape (N,), lower and upper bounds

    Testing Code:
    >>> import numpy as np
    >>> from qball.tools.bounds import rice_paramci_batch
    >>> from scipy.stats import rice
    >>> N = 20
    >>> alpha = 0.9
    >>> sigma = 1/20
    >>> mask = np.ones((N,), dtype=bool)
    >>> mask[:5] = False
    >>> for B in [10**i for i in range(5)]:
    >>>     ground_truth = np.random.uniform(size=(N,))
    >>>     ground_truth[np.logical_not(mask)] = 0.5
    >>>     data = rice.rvs(ground_truth[None,:]/sigma, scale=sigma, size=(B,N))
    >>>     sigma = rice_sigma_mle(data, mask)
    >>>     data_nu, data_l, data_u = rice_nu_paramci_batch(data, sigma, alpha)
    >>>     lower_dist = np.fmax(0, data_l - ground_truth)
    >>>     upper_dist = np.fmax(0, ground_truth - data_u)
    >>>     dist = np.sqrt(np.sum(lower_dist**2 + upper_dist**2))
    >>>     print("Distance for B=%05d: %.2f" % (B, dist))
    """
    b_batch, n_values  = data.shape

    # Compute confidence intervals using the likelihood ratio test (LRT)
    data_l = np.zeros(n_values)
    data_u = np.zeros(n_values)
    thresh = chi2.ppf(1.0-alpha, 1)
    paramci_partial = partial(rice_nu_paramci, sigma=sigma, thresh=thresh)

    # parallelize using all available CPU cores
    p = Pool(processes=None)
    res = p.map(paramci_partial, data.T)
    p.terminate()
    data_nu, data_l, data_u = np.array(res).T

    return data_nu, data_l, data_u

def rice_sigma_mle(data, mask, max_samples=1000):
    """  Estimate scaling parameter of the Rice distribution from background

    Args:
        data : numpy array of shape (B,N)
            N values, B iid noisy samples each
        mask : numpy array of shape (N,) containing 1s and 0s (foreground mask)
        max_samples : maximum number of samples used for performance reasons

    Returns:
        estimate of scaling parameter sigma
    """
    b_batch, n_values  = data.shape
    n_samples = min(b_batch*np.sum(np.logical_not(mask)), max_samples)
    samples = data[:,np.logical_not(mask)].ravel()[0:n_samples]
    _, _, sigma = rice.fit(samples, floc=0)
    logging.info('Estimated sigma=%.5f from n=%d samples.', sigma, n_samples)
    return sigma

def compute_hardi_bounds(data, alpha, mask=None):
    """ Compute fidelity bounds for (subdomain) of HARDI signal `data`.

    Args:
        data : dict, data description
            data['raw'] : numpy array of shape (B,X,Y,Z,L)
                three-dim. (B-batch of) HARDI signals with L b-vectors
            data['slice'] : indexing into (X,Y,Z) of raw data
                bounds are only computed for this slice
            data['b_sph'] : Sphere object from b-vectors
        alpha : confidence level
        mask : numpy array of shape (X,Y,Z)
            foreground mask, containing only 1s and 0s

    Returns:
        fl, fu : numpy arrays of shape (L,N)
            lower and upper bounds for averaged log(-log(data))
            the image dimensions are raveled
    """
    b_sph = data['b_sph']
    data_raw = data['raw']
    if len(data_raw.shape) < 5:
        data_raw = data_raw[None]

    b_batch = data_raw.shape[0]
    l_labels = b_sph.mdims['l_labels']
    assert(data_raw.shape[-1] == l_labels)

    if mask is None:
        # automatically estimate foreground from histogram thresholding (Otsu)
        mask = np.mean(data_raw, axis=(0,-1))
        thresh = otsu(mask)
        mask = (mask <= thresh)
        if len(np.squeeze(mask).shape) == 2:
            logging.debug("\n%s" % matrix2brl(np.squeeze(mask).astype(int)))
    mask = np.tile(mask, (1,1,1,l_labels)).ravel()
    sigma = rice_sigma_mle(data_raw.reshape(b_batch,-1), mask)

    logging.info('Computing confidence intervals with confidence level %.3f'
        ' from batch of size %d...', alpha, b_batch)
    data_sliced = data_raw[(slice(None),) + data['slice']]
    imagedims = data_sliced.shape[1:-1]
    n_image = np.prod(imagedims)
    data_flat = data_sliced.reshape(b_batch,-1)
    _, data_l, data_u = rice_nu_paramci_batch(data_flat, sigma, alpha)

    # Postprocessing of bounds
    clip_hardi_data(data_l)
    clip_hardi_data(data_u)
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
