
import os, shutil
import numpy as np

from compute_dists import l2_dist, compute_dists
from optimize_param import ParamOptimizer

import logging

class LambdaOptimizer(ParamOptimizer):
    par_step = 1.0
    par_min = 0.0
    par_key = lambda self,par: "%.4f" % par
    par_name = "lbd"
    tol = 4e-2

    def run(self):
        if self.result is None:
            self.compute(self.par_step)
        ParamOptimizer.run(self)

    def compute(self, lbd):
        output_dir = self.dirname(self.par_key(lbd))
        run_exp = self.resume

        if not os.path.exists(output_dir):
            shutil.copytree(self.basedir, output_dir,
                            ignore=shutil.ignore_patterns("*.log","*.zip"))
            run_exp = True

        if run_exp:
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
        if self.par_key(0.0) in self.fulldists:
            dists_npz['noise_%s' % distname] = self.fulldists[self.par_key(0.0)]

        dists_npz = compute_dists(output_dir, self.dist, verbose=False,
                                  precomputed=dists_npz, redist=self.redist)

        if dists_npz is None:
            # something went wrong, recompute
            shutil.rmtree(output_dir)
            self.compute(lbd)

        if self.par_key(0.0) not in self.fulldists:
            d = dists_npz['noise_%s' % distname]
            self.fulldists[self.par_key(0.0)] = d
            d_sum = np.sum(d)
            logging.info("Noise: %s=%.5f (min: %.5f, max: %.5f)" % \
                (distname, d_sum, np.amin(d), np.amax(d)))
            self.dists[self.par_key(0.0)] = d_sum

        d = dists_npz[distname]
        self.fulldists[self.par_key(lbd)] = d
        d_sum = np.sum(d)
        logging.info("%s=%s: %s=%.5f (min: %.5f, max: %.5f)" % \
            (self.par_name, self.par_key(lbd), distname, d_sum, np.amin(d), np.amax(d)))
        self.dists[self.par_key(lbd)] = d_sum

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
    parser.add_argument('--dist', metavar='DIST', type=str, default="l2")
    parser.add_argument('--cvx', action="store_true", default=False)
    parsed_args = parser.parse_args()

    import importlib
    exp = importlib.import_module("qball.experiments.%s" % parsed_args.experiment)

    distfun = l2_dist
    if parsed_args.dist == "w1":
        from qball.tools.w1dist import w1_dist
        distfun = w1_dist
    elif parsed_args.dist == "kl":
        from compute_dists import kl_dist
        distfun = kl_dist

    basedir = None if parsed_args.basedir == "" else parsed_args.basedir
    logging.info("==> Optimizing lambda for dist '%s' and basedir '%s'..." % \
          (parsed_args.dist, basedir))
    opt = LambdaOptimizer(exp.MyExperiment, parsed_args.model, basedir=basedir,
                          dist=distfun, resume=parsed_args.resume,
                          redist=parsed_args.redist, cvx=parsed_args.cvx,
                          params=parsed_args.params)
    opt.run()
    logging.info("==> Optimal lambda for dist '%s' and basedir '%s': %.4f" % \
          (parsed_args.dist, basedir, opt.result))
    logging.info("")
