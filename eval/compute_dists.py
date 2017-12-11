
import sys, os, pickle
import numpy as np

import dipy.core.sphere
from dipy.reconst.shm import CsaOdfModel

try:
    import qball
except:
    import set_qball_path
from qball.sphere import load_sphere
from qball.tools import normalize_odf

def kl_dist(f1, f2, b_sph):
    """Entropy or Kullback-Leibler divergence"""
    f1_pos = np.fmax(np.spacing(1),f1)
    f2_pos = np.fmax(np.spacing(1),f2)
    return np.einsum('k,ki->i', b_sph.b, f1_pos*np.log(f1_pos/f2_pos))

def l2_dist(f1, f2, b_sph):
    return np.sqrt(np.einsum('k,ki->i', b_sph.b, (f1 - f2)**2))

def load_b_sph(output_dir):
    gtab_file = os.path.join(output_dir, 'gtab.pickle')
    gtab = pickle.load(open(gtab_file, 'rb'))
    b_vecs = gtab.bvecs[gtab.bvals > 0,...]
    return load_sphere(vecs=b_vecs.T)

def reconst_f(output_dir, b_sph=None):
    params_file = os.path.join(output_dir, 'params.pickle')
    gtab_file = os.path.join(output_dir, 'gtab.pickle')
    S_data_orig_file = os.path.join(output_dir, 'S_data_orig.np')
    S_data_file = os.path.join(output_dir, 'S_data.np')

    baseparams = pickle.load(open(params_file, 'rb'))
    gtab = pickle.load(open(gtab_file, 'rb'))
    S_data = np.load(open(S_data_file, 'rb'))

    S_data_list = [S_data]
    try:
        S_data_orig = np.load(open(S_data_orig_file, 'rb'))
        S_data_list.append(S_data_orig)
    except:
        pass

    l_labels = np.sum(gtab.bvals > 0)
    imagedims = S_data.shape[:-1]
    b_vecs = gtab.bvecs[gtab.bvals > 0,...]
    if b_sph is None:
        b_sph = load_sphere(vecs=b_vecs.T)
    qball_sphere = dipy.core.sphere.Sphere(xyz=b_vecs)
    basemodel = CsaOdfModel(gtab, **baseparams['base'])
    fs = []
    for S in S_data_list:
        f = basemodel.fit(S).odf(qball_sphere)
        f = np.clip(f, 0, np.max(f, -1)[..., None])
        f = np.array(f.reshape(-1, l_labels).T, order='C')
        normalize_odf(f, b_sph.b)
        fs.append(f)
    return tuple(fs)

def compute_dists(output_dir, distfun,
                  precomputed=None, redist=False, verbose=True):
    dists_file = os.path.join(output_dir, 'dists.npz')
    result_file = os.path.join(output_dir, 'result_raw.pickle')

    if not os.path.exists(result_file):
        print("No results found, skipping...")
        return
    else:
        result = pickle.load(open(result_file, 'rb'))[0]

    try:
        upd = result['u']
    except KeyError:
        upd = result['u1']

    b_sph = load_b_sph(output_dir)
    reconst = reconst_f(output_dir, b_sph)
    if len(reconst) == 2:
        f_gt, f_noisy = reconst
    else:
        f_noisy = reconst[0]
        f_gt = f_noisy.copy()

    l_labels = upd.shape[0]
    upd = np.array(upd.reshape(l_labels, -1), order='C')

    if not os.path.exists(dists_file):
        dists_npz = {}
    else:
        dists_npz = dict(np.load(open(dists_file, 'rb')))

    if precomputed is not None:
        dists_npz.update(precomputed)
    else:
        precomputed = {}

    distname = distfun.__name__
    distname_noise = 'noise_%s' % distname

    if distname_noise not in precomputed:
        if redist or distname_noise not in dists_npz:
            d = distfun(f_gt, f_noisy, b_sph)
            dists_npz[distname_noise] = d

    if distname not in precomputed:
        if redist or distname not in dists_npz:
            d = distfun(f_gt, upd, b_sph)
            dists_npz[distname] = d

    if verbose:
        d = dists_npz[distname_noise]
        d_sum = np.sum(d)
        print("Noise: %.5f (min: %.5f, max: %.5f)" % (
            d_sum, np.amin(d), np.amax(d)))

        d = dists_npz[distname]
        d_sum = np.sum(d)
        print("Result: %.5f (min: %.5f, max: %.5f)" % (
            d_sum, np.amin(d), np.amax(d)))

    np.savez_compressed(dists_file, **dists_npz)
    return dists_npz

if __name__ == "__main__":
    distfun = l2_dist
    basedirs = sys.argv[1:]
    if sys.argv[1] == "w1":
        from qball.tools.w1dist import w1_dist
        distfun = w1_dist
        basedirs = sys.argv[2:]
    elif sys.argv[1] == "kl": # Kullback-Leibler or entropy
        distfun = kl_dist
        basedirs = sys.argv[2:]
    elif sys.argv[1] == "peak":
        from compute_peaks import peak_dist
        distfun = peak_dist
        basedirs = sys.argv[2:]

    for i,output_dir in enumerate(basedirs):
        print("=> %s" % output_dir)
        compute_dists(output_dir, distfun, verbose=True)
        if i+1 < len(basedirs):
            print("")
