
import matplotlib
matplotlib.use('Agg')

from manifold_sphere import load_sphere
from eval_w1dist import w1_dist
from tools import normalize_odf

import sys, os, pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import dipy.core.sphere
from dipy.reconst.shm import CsaOdfModel

noise = True
for output_dir in sys.argv[1:]:
    gtab_file = os.path.join(output_dir, 'gtab.pickle')
    S_data_file = os.path.join(output_dir, 'S_data.np')
    S_data_orig_file = os.path.join(output_dir, 'S_data_orig.np')
    pd_result_file = os.path.join(output_dir, 'result_raw.npz')
    dists_file = os.path.join(output_dir, 'dists.npz')
    dists_plot_file = os.path.join(output_dir, 'plot_dists.pdf')

    gtab = pickle.load(open(gtab_file, 'rb'))
    S_data = np.load(open(S_data_file, 'rb'))
    S_data_orig = np.load(open(S_data_orig_file, 'rb'))
    upd = np.load(open(pd_result_file, 'rb'))['arr_0']

    l_labels = S_data.shape[-1]
    imagedims = S_data.shape[:-1]

    b_vecs = gtab.bvecs[gtab.bvals > 0,...]
    qball_sphere = dipy.core.sphere.Sphere(xyz=b_vecs)
    b_sph = load_sphere(vecs=b_vecs.T)

    baseparams = {
        'sh_order': 6,
        'smooth': 0,
        'min_signal': 0,
        'assume_normed': True
    }
    basemodel = CsaOdfModel(gtab, **baseparams)
    f = basemodel.fit(S_data).odf(qball_sphere)
    fin = np.clip(f, 0, np.max(f, -1)[..., None])
    f = basemodel.fit(S_data_orig).odf(qball_sphere)
    fin_orig = np.clip(f, 0, np.max(f, -1)[..., None])

    fin = np.array(fin.reshape(-1, l_labels).T, order='C')
    fin_orig = np.array(fin_orig.reshape(-1, l_labels).T, order='C')
    upd = upd.reshape(l_labels, -1)

    normalize_odf(fin, b_sph.b)
    normalize_odf(fin_orig, b_sph.b)
    normalize_odf(upd, b_sph.b)

    try:
        npzfile = np.load(open(dists_file, 'rb'))
        if noise:
            l2d_noise = npzfile['l2d_noise']
            w1d_noise = npzfile['w1d_noise']
        w1d_upd = npzfile['w1d_upd']
        l2d_upd = npzfile['l2d_upd']
    except:
        if noise:
            l2d_noise = np.einsum('ki,ki->i', fin_orig - fin, fin_orig - fin)
            w1d_noise = w1_dist(fin_orig, fin, b_sph)
        w1d_upd = w1_dist(fin_orig, upd, b_sph)
        l2d_upd = np.einsum('ki,ki->i', fin_orig - upd, fin_orig - upd)
        np.savez(open(dists_file, 'wb'),
            l2d_noise=l2d_noise, w1d_noise=w1d_noise,
            w1d_upd=w1d_upd, l2d_upd=l2d_upd
        )

    if noise:
        print("Noise (W1): %.5f (min: %.5f, max: %.5f)" % (
            np.sum(w1d_noise), np.amin(w1d_noise), np.amax(w1d_noise)))
        print("Noise (L2): %.5f (min: %.5f, max: %.5f)" % (
            np.sum(l2d_noise), np.amin(l2d_noise), np.amax(l2d_noise)))
        noise = False

    print(output_dir)
    print("Res (W1): %.5f (min: %.5f, max: %.5f)" % (
        np.sum(w1d_upd), np.amin(w1d_upd), np.amax(w1d_upd)))
    print("Res (L2): %.5f (min: %.5f, max: %.5f)" % (
        np.sum(l2d_upd), np.amin(l2d_upd), np.amax(l2d_upd)))

    fig = plt.figure()
    dist_arr = [
        (w1d_noise, "W1 distance (noise)"),
        (l2d_noise, "L2 distance (noise)"),
        (w1d_upd,   "W1 distance (result)"),
        (l2d_upd,   "L2 distance (result)")
    ]
    subplot_opts = {
        'xticklabels': [],
        'yticklabels': [],
    }
    for i, (dist, descr) in enumerate(dist_arr):
        a = fig.add_subplot(2,2,i+1, **subplot_opts)
        if len(imagedims) == 1:
            d = np.atleast_2d(dist.reshape(imagedims))
        else:
            d = dist.reshape(imagedims).T
        plt.imshow(d, cmap=cm.coolwarm, origin='lower', interpolation='none')
        a.set_title(descr)
    plt.savefig(dists_plot_file)
    #plt.show()
