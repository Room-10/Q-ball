
import sys, os, pickle, shutil
import numpy as np

from compute_dists import l2_dist, load_b_sph, reconst_f, compute_dists

try:
    import qball
except:
    import set_qball_path
from qball.tools import normalize_odf

def lbd_key(lbd):
    return "%.4f" % lbd

class LambdaOptimizer(object):
    def __init__(self, experiment, model, basedir=None, dist=l2_dist,
                 resume=False, redist=False, cvx=False, params=""):
        self.experiment = experiment
        self.model = model
        if basedir is not None:
            self.basedir = basedir.rstrip("/")
        else:
            exp_args = [self.model, '--batch']
            exp = self.experiment(exp_args)
            exp.load_imagedata()
            self.basedir = exp.output_dir
        self.dist = dist
        self.resume = resume
        self.redist = redist
        self.cvx = cvx
        self.params = params

        self.result = None
        self.dists = {}
        self.fulldists = {}
        self.b_sph = load_b_sph(self.basedir)

    def run(self):
        # ----------------------------------------------------------------------
        #   init
        # ----------------------------------------------------------------------
        lbd_l = lbd = lbd_r = 0
        for i in np.arange(1, 20, dtype=np.float64):
            self.dists[lbd_key(i)] = self.compute(i)
            relgap = (self.dists[lbd_key(i-1)] - self.dists[lbd_key(i)])/self.dists[lbd_key(i)]
            if relgap < 1e-3:
                if i == 1.0:
                    lbd_l = 0.0
                    lbd = 0.5
                    lbd_r = 1.0
                    self.dists[lbd_key(lbd)] = self.compute(lbd)
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

    def compute(self, lbd):
        output_dir = self.basedir
        if self.basedir[-len(self.model):] != self.model:
            output_dir = "%s-%s" % (output_dir, self.model)
        output_dir = "%s-%s" % (output_dir, lbd_key(lbd))

        if not os.path.exists(output_dir):
            shutil.copytree(self.basedir, output_dir,
                            ignore=shutil.ignore_patterns("*.log","*.zip"))

        exp_params = 'lbd=%f' % lbd
        if len(self.params) > 0:
            exp_params = '%s,%s' % (exp_params, self.params)
        exp_args = [self.model, '--output', output_dir, '--plot','no']
        exp_args += ['--params', exp_params]
        if self.resume:
            exp_args.append('--resume')
        if self.cvx:
            exp_args.append('--cvx')
        exp = self.experiment(exp_args)
        exp.run()

        distname = self.dist.__name__
        dists_npz = {}
        if lbd_key(0.0) in self.fulldists:
            dists_npz['noise_%s' % distname] = self.fulldists[lbd_key(0.0)]

        dists_npz = compute_dists(output_dir, self.dist, verbose=False,
                                  precomputed=dists_npz, redist=self.redist)

        if lbd_key(0.0) not in self.fulldists:
            d = dists_npz['noise_%s' % distname]
            self.fulldists[lbd_key(0.0)] = d
            d_sum = np.sum(d)
            print("Noise: %.5f (min: %.5f, max: %.5f)" % (
                d_sum, np.amin(d), np.amax(d)))
            self.dists[lbd_key(0.0)] = d_sum

        d = dists_npz[distname]
        self.fulldists[lbd_key(lbd)] = d
        d_sum = np.sum(d)
        print("%s: %.5f (min: %.5f, max: %.5f)" % (
            lbd_key(lbd), d_sum, np.amin(d), np.amax(d)))
        return d_sum

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Optimize regularization parameter.")
    parser.add_argument('experiment', metavar='EXPERIMENT', type=str)
    parser.add_argument('model', metavar='MODEL', type=str)
    parser.add_argument('--basedir', metavar='BASENAME', type=str, default="")
    parser.add_argument('--params', metavar='PARAMS', type=str, default="")
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--redist', action="store_true", default=False,
                        help="Recalculate distances.")
    parser.add_argument('--w1', action="store_true", default=False)
    parser.add_argument('--cvx', action="store_true", default=False)
    parsed_args = parser.parse_args()

    import importlib
    exp = importlib.import_module("qball.experiments.%s" % parsed_args.experiment)

    distfun = l2_dist
    if parsed_args.w1:
        from qball.tools.w1dist import w1_dist
        distfun = w1_dist
    basedir = None if parsed_args.basedir == "" else parsed_args.basedir
    opt = LambdaOptimizer(exp.MyExperiment, parsed_args.model, basedir=basedir,
                          dist=distfun, resume=parsed_args.resume,
                          redist=parsed_args.redist, cvx=parsed_args.cvx,
                          params=parsed_args.params)
    opt.run()
    print(opt.result)
