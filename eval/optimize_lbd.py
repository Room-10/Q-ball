
import sys, os, pickle, shutil

import numpy as np

import dipy.core.sphere
from dipy.reconst.shm import CsaOdfModel

try:
    import qball
except:
    import set_qball_path
from qball.sphere import load_sphere
from qball.tools import normalize_odf

def lbd_key(lbd):
    return "%.4f" % lbd

def l2_dist(f1, f2):
    return np.sqrt(np.einsum('ki,ki->i', f1 - f2, f1 - f2))

class LambdaOptimizer(object):
    def __init__(self, basedir, experiment, dist=l2_dist, resume=False, redist=False):
        self.basedir = basedir.rstrip("/")
        self.dist = dist
        self.experiment = experiment
        self.resume = resume
        self.redist = redist
        self.result = None
        self.dists = {}
        self.fulldists = {}
        self.load_data()

    def run(self):
        # ----------------------------------------------------------------------
        #   init
        # ----------------------------------------------------------------------
        d = self.dist(self.f_gt, self.f_noisy)
        self.fulldists[lbd_key(0.0)] = d
        d_sum = np.sum(d)
        print("Noise: %.5f (min: %.5f, max: %.5f)" % (
            d_sum, np.amin(d), np.amax(d)))
        self.dists[lbd_key(0.0)] = d_sum

        lbd_l = lbd = lbd_r = 0
        for i in np.arange(1, 20, dtype=np.float64):
            self.dists[lbd_key(i)] = self.compute(i)
            relgap = (self.dists[lbd_key(i-1)] - self.dists[lbd_key(i)])/self.dists[lbd_key(i)]
            if relgap < 1e-3:
                if i == 1.0:
                    lbd_l = 0.0
                    lbd = 0.5
                    lbd_r = 1.0
                else:
                    lbd_l = i-2.0
                    lbd = i-1.0
                    lbd_r = i
                break

        if self.dists[lbd_key(lbd)] < self.dists[lbd_key(lbd_r)] \
           and self.dists[lbd_key(lbd)] < self.dists[lbd_key(lbd_l)]:
            # ------------------------------------------------------------------
            #   bisect
            # ------------------------------------------------------------------
            relgap = (lbd_r - lbd_l)/lbd
            next = "l"
            while relgap > 4e-2:
                if next == "l":
                    lbd_new = (lbd_l + lbd)/2
                    self.dists[lbd_key(lbd_new)] = self.compute(lbd_new)
                    if self.dists[lbd_key(lbd_new)] < self.dists[lbd_key(lbd)]:
                        lbd_l, lbd, lbd_r = lbd_l, lbd_new, lbd
                    else:
                        lbd_l = lbd_new
                    next = "r"
                else:
                    lbd_new = (lbd + lbd_r)/2
                    self.dists[lbd_key(lbd_new)] = self.compute(lbd_new)
                    if self.dists[lbd_key(lbd_new)] < self.dists[lbd_key(lbd)]:
                        lbd_l, lbd, lbd_r = lbd, lbd_new, lbd_r
                    else:
                        lbd_r = lbd_new
                    next = "l"
                relgap = (lbd_r - lbd_l)/lbd
        else:
            lbd = lbd_r
        self.result = lbd

    def load_data(self):
        params_file = os.path.join(self.basedir, 'params.pickle')
        gtab_file = os.path.join(self.basedir, 'gtab.pickle')
        S_data_file = os.path.join(self.basedir, 'S_data.np')
        S_data_orig_file = os.path.join(self.basedir, 'S_data_orig.np')

        self.baseparams = pickle.load(open(params_file, 'rb'))
        self.gtab = pickle.load(open(gtab_file, 'rb'))
        self.S_data = np.load(open(S_data_file, 'rb'))
        self.S_data_orig = np.load(open(S_data_orig_file, 'rb'))
        self.reconst_f()

    def reconst_f(self):
        l_labels = self.S_data.shape[-1]
        imagedims = self.S_data.shape[:-1]
        b_vecs = self.gtab.bvecs[self.gtab.bvals > 0,...]
        self.qball_sphere = dipy.core.sphere.Sphere(xyz=b_vecs)
        self.b_sph = load_sphere(vecs=b_vecs.T)
        basemodel = CsaOdfModel(self.gtab, **self.baseparams['base'])
        fs = []
        for S in [self.S_data, self.S_data_orig]:
            f = basemodel.fit(S).odf(self.qball_sphere)
            f = np.clip(f, 0, np.max(f, -1)[..., None])
            f = np.array(f.reshape(-1, l_labels).T, order='C')
            normalize_odf(f, self.b_sph.b)
            fs.append(f)
        self.f_noisy, self.f_gt = fs

    def compute(self, lbd):
        output_dir = "%s-%s" % (self.basedir, lbd_key(lbd))

        if not os.path.exists(output_dir):
            shutil.copytree(self.basedir, output_dir)
            params = dict(self.baseparams)
            params['fit']['solver_params']['lbd'] = lbd
            params_file = os.path.join(output_dir, 'params.pickle')
            pickle.dump(params, open(params_file, 'wb'))

        exp_args = [output_dir, '--batch']
        if self.resume:
            exp_args.append('--resume')
        exp = self.experiment(exp_args)
        exp.run()

        l_labels = exp.upd.shape[-1]
        f_lbd = np.array(exp.upd.reshape(-1, l_labels).T, order='C')
        normalize_odf(f_lbd, self.b_sph.b)

        dist_file = os.path.join(output_dir, 'dists.npz')
        distname = self.dist.__name__
        distname_noise = 'noise_%s' % distname
        if not os.path.exists(dist_file):
            dist_npz = {}
        else:
            dist_npz = dict(np.load(open(dist_file, 'rb')))

        if distname_noise not in dist_npz:
            dist_npz[distname_noise] = self.fulldists[lbd_key(0.0)]

        if self.redist or distname not in dist_npz:
            dist_npz[distname] = self.dist(self.f_gt, f_lbd)

        np.savez_compressed(dist_file, **dist_npz)
        d = dist_npz[distname]
        self.fulldists[lbd_key(lbd)] = d
        d_sum = np.sum(d)
        print("%s: %.5f (min: %.5f, max: %.5f)" % (
            lbd_key(lbd), d_sum, np.amin(d), np.amax(d)))
        return d_sum

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Optimize for regularization parameter.")
    parser.add_argument('basedir', metavar='BASENAME', type=str)
    parser.add_argument('demo', metavar='DEMO', type=str)
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--redist', action="store_true", default=False,
                        help="Recalculate distances.")
    parser.add_argument('--w1', action="store_true", default=False)
    parsed_args = parser.parse_args()

    import importlib
    exp = importlib.import_module("demos.%s" % parsed_args.demo)
    opt = LambdaOptimizer(parsed_args.basedir, exp.MyExperiment,
                          resume=parsed_args.resume, redist=parsed_args.redist)
    if parsed_args.w1:
        from qball.tools.w1dist import w1_dist as __w1_dist
        def w1_dist(f1, f2):
            return __w1_dist(f1, f2, opt.b_sph)
        opt.dist = w1_dist
    opt.run()
    print(opt.result)
