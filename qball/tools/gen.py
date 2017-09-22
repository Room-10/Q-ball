
import os, pickle, warnings, logging
import numpy as np

import nibabel as nib
from dipy.core.gradients import GradientTable, gradient_table
from dipy.sims.voxel import multi_tensor
from dipy.sims.phantom import add_noise
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import _make_fetcher, dipy_home
from dipy.io.gradients import read_bvals_bvecs
# median filter -> otsu -> mask (bg blacked out)
from dipy.segment.mask import median_otsu

from qball.sphere import load_sphere
from qball.phantom import FiberPhantom

try:
    from contextlib import redirect_stdout
except ImportError:
    # shim for Python 2.x
    import sys
    from contextlib import contextmanager
    @contextmanager
    def redirect_stdout(new_target):
        old_target, sys.stdout = sys.stdout, new_target # replace sys.stdout
        try:
            yield new_target # run some code with the replaced stdout
        finally:
            sys.stdout = old_target # restore to the previous value

from dipy.reconst.csdeconv import recursive_response
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy

def csd_response(gtab, data):
    """ Estimate response function for given HARDI data.
    Unfortunately, does not work for 2d (synthetic) data (why?).
    """
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=data[..., 0] > 200)
    FA = fractional_anisotropy(tenfit.evals)
    MD = dti.mean_diffusivity(tenfit.evals)
    wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))
    response = recursive_response(gtab, data, mask=wm_mask, sh_order=8,
                      peak_thr=0.01, init_fa=0.08, init_trace=0.0021, iter=8,
                      convergence=0.001, parallel=True)
    return response

def synth_isbi2013(snr=30):
    supported_snrs = np.array([10,20,30])
    snr = supported_snrs[np.argmin(np.abs(supported_snrs - snr))]
    with redirect_stdout(open(os.devnull, "w")), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img, gtab = read_isbi2013_challenge(snr=snr)
        assert(img.shape[-1] == gtab.bvals.size)
    S_data = np.array(img[12:27,22,21:36], order='C')
    return S_data.copy(), S_data, gtab

def rw_stanford(snr=None, csd=False):
    with redirect_stdout(open(os.devnull, "w")), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img, gtab = read_stanford_hardi()
        assert(img.shape[-1] == gtab.bvals.size)
        data = img.get_data()

    if False:
        # from dipy qbi-csa example (used for SSVM):
        # http://nipy.org/dipy/examples_built/reconst_csa.html
        maskdata, mask = median_otsu(data, median_radius=3, numpass=1,
            autocrop=True, vol_idx=range(10, 50), dilate=2)
        S_data = np.array(maskdata[13:43, 44:74, 28], order='C')
    else:
        # from dipy csd example:
        # http://nipy.org/dipy/examples_built/reconst_csd.html
        S_data = np.array(data[20:50, 55:85, 38], order='C')

    S_data_orig = S_data.copy()
    if snr is not None:
        S_data[:] = add_noise(S_data_orig, snr=snr)

    if csd:
        resp_file = "cache/resp-stanford_hardi.pickle"
        try:
            resp = pickle.load(open(resp_file, 'rb'))
        except:
            logging.info("No cached response function, estimating...")
            resp = csd_response(gtab, data)
            pickle.dump(resp, open(resp_file, 'wb'))
    else:
        resp = None

    return S_data_orig, S_data, gtab, resp

def synth_unimodals(bval=3000, imagedims=(8,), jiggle=10, snr=None):
    d_image = len(imagedims)
    n_image = np.prod(imagedims)

    sph = load_sphere(refinement=2)
    l_labels = sph.mdims['l_labels']
    gtab = GradientTable(bval * sph.v.T, b0_threshold=0)

    S_data_orig = np.stack([one_fiber_signal(gtab, 0, snr=None)]*n_image) \
                    .reshape(imagedims + (l_labels,))

    S_data = np.stack([
        one_fiber_signal(gtab, 0+r, snr=snr)
        for r in jiggle*np.random.randn(n_image)
    ]).reshape(imagedims + (l_labels,))

    return S_data_orig, S_data, gtab

def synth_unimodals_linear(bval=3000, imagedims=(12,)):
    d_image = len(imagedims)
    n_image = np.prod(imagedims)

    sph = load_sphere(refinement=2)
    l_labels = sph.mdims['l_labels']
    gtab = GradientTable(bval * sph.v.T, b0_threshold=0)

    S_data_orig = np.stack([
        one_fiber_signal(gtab, r, snr=None, eval_factor=15)
        for r in np.linspace(-45, 45, n_image)
    ]).reshape(imagedims + (l_labels,))

    return S_data_orig, S_data_orig.copy(), gtab

def synth_bimodals(bval=3000, const_width=5, snr=None):
    imagedims = (const_width*2,)
    d_image = len(imagedims)
    n_image = np.prod(imagedims)

    sph = load_sphere(refinement=2)
    l_labels = sph.mdims['l_labels']
    gtab = GradientTable(bval * sph.v.T, b0_threshold=0)

    S_data = np.stack(
        [two_fiber_signal(gtab, [0,70], snr=None)]*const_width
        + [uniform_signal(gtab, snr=None)]*const_width
    ).reshape(imagedims + (l_labels,))

    S_data_orig = S_data.copy()
    if snr is not None:
        S_data[:] = add_noise(S_data_orig, snr=snr)
    return S_data_orig, S_data, gtab

def synth_cross(res=15, snr=None):
    f1 = lambda x: 0.5*(x + 0.3)**3 + 0.05
    f1inv = lambda y: (y/0.5)**(1/3) - 0.3
    f2 = lambda x: 0.7*(1.5 - x)**3 - 0.5
    f2inv = lambda y: 1.5 - ((y + 0.5)/0.7)**(1/3)

    p = FiberPhantom(res)
    p.add_curve(lambda t: (t,f1(t)), tmin=-0.2, tmax=f1inv(1.0)+0.2)
    p.add_curve(lambda t: (t,f2(t)), tmin=f2inv(1.0)-0.2, tmax=f2inv(0.0)+0.2)

    gtab, S_data = p.gen_hardi(snr=snr)
    _, S_data_orig = p.gen_hardi(snr=None)

    return S_data_orig, S_data, gtab, p

# 64 directions
# 50x50x50 voxels
fetch_isbi2013_challenge = _make_fetcher(
    "fetch_isbi2013_challenge",
    os.path.join(dipy_home, 'isbi2013_challenge'),
    'http://hardi.epfl.ch/static/events/2013_ISBI/_downloads/',
    [
        'testing-data_DWIS_hardi-scheme_SNR-10.nii.gz',
        'testing-data_DWIS_hardi-scheme_SNR-20.nii.gz',
        'testing-data_DWIS_hardi-scheme_SNR-30.nii.gz',
        'hardi-scheme.bval', 'hardi-scheme.bvec',
    ],
    [
        'hardi-scheme_SNR-10.nii.gz',
        'hardi-scheme_SNR-20.nii.gz',
        'hardi-scheme_SNR-30.nii.gz',
        'hardi-scheme.bval', 'hardi-scheme.bvec',
    ],
    [
        'c3d97559f418358bb69467a0b5809630',
        '33640b1297c8b498e0328fe268dbd5c1',
        'a508716c5eec555a77a34817acafb0ca',
        '92811d6e800a6a56d7498b0c4b5ed0c2',
        'c8f5025b9d91037edb6cd00af9bd3e41',
    ])

def read_isbi2013_challenge(snr=30):
    """ Load ISBI 2013's HARDI reconstruction challenge dataset

    Returns
    -------
        img : obj, Nifti1Image
        gtab : obj, GradientTable
    """
    files, folder = fetch_isbi2013_challenge()
    fraw = os.path.join(folder, 'hardi-scheme_SNR-%d.nii.gz' % snr)
    fbval = os.path.join(folder, 'hardi-scheme.bval')
    fbvec = os.path.join(folder, 'hardi-scheme.bvec')
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    img = nib.load(fraw).get_data()
    # make sure the bvecs are only for the half sphere
    dists = []
    pairs = []
    for i,v in enumerate(bvecs):
        for j,w in enumerate(bvecs[(i+1):,:]):
            pairs.append((i,i+1+j))
            dists.append(np.sum((v+w)**2))
    assert(np.min(dists) > 1e-4)
    # for each bvec, add the opposite bvec
    img = np.concatenate((img, img[...,bvals > 0]), axis=-1)
    bvecs = np.concatenate((bvecs, -bvecs[bvals > 0]), axis=0)
    bvals = np.concatenate((bvals, bvals[bvals > 0]), axis=0)
    gtab = gradient_table(bvals, bvecs)
    return img, gtab

def uniform_signal(gtab, snr=None):
    mevals = np.array([[300e-6, 300e-6, 300e-6]])
    signal, sticks = multi_tensor(gtab, mevals,
        S0=1., angles=[(90,0)], fractions=[100], snr=snr)
    return signal

def one_fiber_signal(gtab, angle, snr=None, eval_factor=5):
    eval_base = 300e-6
    mevals = np.array([[eval_factor*eval_base, eval_base, eval_base]])
    signal, sticks = multi_tensor(gtab, mevals,
        S0=1., angles=[(90,angle)], fractions=[100], snr=snr)
    return signal

def two_fiber_signal(gtab, angles, snr=None):
    mevals = np.array([[1800e-6, 200e-6, 200e-6]]*2)
    signal, sticks = multi_tensor(gtab, mevals,
        S0=1., angles=[(90,angles[0]), (90,angles[1])], fractions=[50,50], snr=snr)
    return signal
