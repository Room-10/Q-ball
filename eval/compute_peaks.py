
import sys, os, pickle

from math import *
import numpy as np
from numpy.linalg import norm
from scipy import stats

import nibabel as nib
from dipy.core.ndindex import ndindex
from dipy.direction.peaks import peak_directions
from dipy.core.sphere import Sphere
from dipy.reconst.shm import CsaOdfModel

try:
    import qball
except:
    import set_qball_path
from qball.sphere import load_sphere
from qball.tools import normalize_odf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_peaks(fname, peaks):
    imagedims = peaks.shape[:-1]

    fig = plt.figure(figsize=(7,7))
    subplot_opts = {
        'aspect': 'equal',
        'xticklabels': [],
        'yticklabels': [],
        'xticks': [],
        'yticks': [],
        'xlim': [0.0,1.0],
        'ylim': [0.0,1.0],
    }
    ax = fig.add_subplot(111, **subplot_opts)

    delta = 1.0/max(imagedims)
    peak_scaling = 0.5*delta
    peaks = peaks.reshape(imagedims + (peaks.shape[-1],))
    for x in range(0,imagedims[0]):
        for y in range(0,imagedims[1]):
            mid = delta*(np.array([x,y]) + 0.5)
            ax.scatter(mid[0], mid[1], s=3, c='b', linewidths=0)
            for d in range(5) :
                peak = peaks[x,y,range(d*3, d*3+3)]
                f = norm(peak)
                if f > 0 :
                    peak /= f
                    data = np.array([
                        mid[:] - 0.5*peak_scaling*peak[[0,2]],
                        mid[:] + 0.5*peak_scaling*peak[[0,2]]
                   ]).T
                    ax.plot(data[0], data[1], 'r')

    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98,
        wspace=0.03, hspace=0)
    plt.savefig(fname)
    #plt.show()

def reconst_f(output_dir, b_sph):
    params_file = os.path.join(output_dir, 'params.pickle')
    gtab_file = os.path.join(output_dir, 'gtab.pickle')
    S_data_file = os.path.join(output_dir, 'S_data.np')

    baseparams = pickle.load(open(params_file, 'rb'))
    gtab = pickle.load(open(gtab_file, 'rb'))
    S_data = np.load(open(S_data_file, 'rb'))

    l_labels = np.sum(gtab.bvals > 0)
    imagedims = S_data.shape[:-1]
    b_vecs = gtab.bvecs[gtab.bvals > 0,...]
    qball_sphere = Sphere(xyz=b_vecs)
    basemodel = CsaOdfModel(gtab, **baseparams['base'])
    f = basemodel.fit(S_data).odf(qball_sphere)
    f = np.clip(f, 0, np.max(f, -1)[..., None])
    f = np.array(f.reshape(-1, l_labels).T, order='C')
    normalize_odf(f, b_sph.b)
    return np.array(f.reshape(l_labels, -1).T, order='C')

def load_b_sph(output_dir):
    gtab_file = os.path.join(output_dir, 'gtab.pickle')
    gtab = pickle.load(open(gtab_file, 'rb'))
    b_vecs = gtab.bvecs[gtab.bvals > 0,...]
    return load_sphere(vecs=b_vecs.T)

def compute_peaks(output_dir, relative_peak_threshold=.5,
                  peak_normalize=1, min_separation_angle=45, max_peak_number=5):
    peaks_file = os.path.join(output_dir, 'peaks.npz')
    result_file = os.path.join(output_dir, 'result_raw.pickle')
    result = pickle.load(open(result_file, 'rb'))[0]

    try:
        u_RECON = result['u']
    except KeyError:
        u_RECON = result['u1']

    b_sph = load_b_sph(output_dir)
    sphere = Sphere(xyz=b_sph.v.T)
    f_noisy = reconst_f(output_dir, b_sph)

    l_labels = u_RECON.shape[0]
    imagedims = u_RECON.shape[1:]
    u_RECON = np.array(u_RECON.reshape(l_labels, -1).T, order='C')

    num_peak_coeffs = max_peak_number * 3
    computed_peaks = []
    for odfs in [u_RECON, f_noisy]:
        peaks = np.zeros(u_RECON.shape[:-1] + (num_peak_coeffs,))
        for index in ndindex(odfs.shape[:-1]):
            vox_peaks, values, _ = peak_directions(odfs[index], sphere,
                                                   float(relative_peak_threshold),
                                                   float(min_separation_angle))

            if peak_normalize == 1:
                values /= values[0]
                vox_peaks = vox_peaks * values[:, None]

            vox_peaks = vox_peaks.ravel()
            m = vox_peaks.shape[0]
            if m > num_peak_coeffs:
                m = num_peak_coeffs
            peaks[index][:m] = vox_peaks[:m]
        computed_peaks.append(peaks.reshape(imagedims + (peaks.shape[-1],)))

    np.savez_compressed(peaks_file, computed_peaks)
    return computed_peaks

def compute_err(peaks, groundtruth):
    niiGT = nib.load(groundtruth)
    niiGT_hdr = niiGT.get_header()
    niiGT_img = niiGT.get_data()[12:27,22,21:36]
    niiGT_dim = niiGT_img.shape
    plot_peaks(os.path.join(output_dir, 'plot_gt_peaks.pdf'), niiGT_img)
    niiRECON_img = peaks
    niiMASK_img = np.ones_like(niiGT_img)
    niiMASK_idx = niiMASK_img==1

    # Correct estimation of the number of fiber compartments
    # Pd : probability of false fibre detection
    # nP : absolute number of compartments that were not detected
    # nM : absolute number of compartments that were falsely detected
    Pd = np.zeros(niiGT_dim[0:3])
    nP = np.zeros(niiGT_dim[0:3])
    nM = np.zeros(niiGT_dim[0:3])

    # Angular precision of the estimated fiber compartments
    # AE : angular error (in degrees) between the estimated fiber directions and the true ones inside the voxel
    AE = np.zeros(niiGT_dim[0:3])

    for x in range(0,niiGT_dim[0]):
        for y in range(0,niiGT_dim[1]):
            # NUMBER OF FIBER POPULATIONS
            #############################

            DIR_true = np.zeros((3,5))
            DIR_est  = np.zeros((3,5))

            # compute M_true, DIR_true, M_est, DIR_est
            M_true = 0
            for d in range(5) :
                dir = niiGT_img[x,y,range(d*3, d*3+3)]
                f = norm(dir)
                if f > 0 :
                    DIR_true[:,M_true] = dir / f
                    M_true += 1
            if M_true == 0 :
                # do not consider this voxel in the final score
                continue    # no fiber compartments found in the voxel

            M_est = 0
            for d in range(5) :
                dir = niiRECON_img[x,y,range(d*3, d*3+3)]
                f = norm(dir)
                if f > 0 :
                    DIR_est[:,M_est] = dir / f
                    M_est += 1

            # compute Pd, nM and nP
            M_diff = M_true - M_est
            Pd[x,y] = 100 * abs(M_diff) / M_true
            if  M_diff > 0 :
                nM[x,y] = M_diff;
            else :
                nP[x,y] = -M_diff

            # ANGULAR ACCURACY
            ##################

            # precompute matrix with angular errors among all estimated and true fibers
            A = np.zeros((M_true, M_est))
            for i in range(0,M_true) :
                for j in range(0,M_est) :
                    err = acos(min(1.0, abs(np.dot(DIR_true[:,i], DIR_est[:,j])))) # crop to 1 for internal precision
                    A[i,j] = min(err, pi-err) / pi * 180;

            # compute the "base" error
            M = min(M_true,M_est)
            err = np.zeros(M)
            notUsed_true = np.array(range(0,M_true))
            notUsed_est  = np.array(range(0,M_est))
            AA = np.copy(A)
            for i in range(0,M) :
                err[i] = np.min(AA)
                r, c = np.nonzero(AA==err[i])
                AA[r[0],:] = float('Inf')
                AA[:,c[0]] = float('Inf')
                notUsed_true = notUsed_true[notUsed_true != r[0]]
                notUsed_est  = notUsed_est[ notUsed_est  != c[0]]

            # account for OVER-ESTIMATES
            if M_true < M_est :
                if M_true > 0:
                    for i in notUsed_est :
                        err = np.append(err, min(A[:,i]))
                else :
                    err = np.append(err, 45)
            # account for UNDER-ESTIMATES
            elif M_true > M_est :
                if M_est > 0:
                    for i in notUsed_true :
                        err = np.append(err, min(A[i,:]))
                else :
                    err = np.append(err, 45)

            AE[x,y] = np.mean(err)

    values = Pd[niiMASK_idx]
    print("Pd, mean   ", np.mean(values))
    print("Pd, std    ", np.std(values))
    print("Pd, min    ", np.min(values))
    print("Pd, 25 perc", stats.scoreatpercentile(values,25))
    print("Pd, 50 perc", np.median(values))
    print("Pd, 75 perc", stats.scoreatpercentile(values,75))
    print("Pd, max    ", np.max(values))

    values = nM[niiMASK_idx]
    print("n-, mean   ", np.mean(values))
    print("n-, std    ", np.std(values))
    print("n-, min    ", np.min(values))
    print("n-, 25 perc", stats.scoreatpercentile(values,25))
    print("n-, 50 perc", np.median(values))
    print("n-, 75 perc", stats.scoreatpercentile(values,75))
    print("n-, max    ", np.max(values))

    values = nP[niiMASK_idx]
    print("n+, mean   ", np.mean(values))
    print("n+, std    ", np.std(values))
    print("n+, min    ", np.min(values))
    print("n+, 25 perc", stats.scoreatpercentile(values,25))
    print("n+, 50 perc", np.median(values))
    print("n+, 75 perc", stats.scoreatpercentile(values,75))
    print("n+, max    ", np.max(values))

    values = AE[niiMASK_idx]
    print("AE, mean   ", np.mean(values))
    print("AE, std    ", np.std(values))
    print("AE, min    ", np.min(values))
    print("AE, 25 perc", stats.scoreatpercentile(values,25))
    print("AE, 50 perc", np.median(values))
    print("AE, 75 perc", stats.scoreatpercentile(values,75))
    print("AE, max    ", np.max(values))

if __name__ == "__main__":
    groundtruth = sys.argv[1]
    basedirs = sys.argv[2:]

    for i,output_dir in enumerate(basedirs):
        print("==> %s" % output_dir)
        u_peaks, f_peaks = compute_peaks(output_dir, relative_peak_threshold=.5,
                  peak_normalize=1, min_separation_angle=25, max_peak_number=5)

        print("-> error of input")
        compute_err(f_peaks, groundtruth)
        plot_peaks(os.path.join(output_dir, 'plot_f_peaks.pdf'), f_peaks)

        print("-> error of restored data set")
        compute_err(u_peaks, groundtruth)
        plot_peaks(os.path.join(output_dir, 'plot_u_peaks.pdf'), u_peaks)
