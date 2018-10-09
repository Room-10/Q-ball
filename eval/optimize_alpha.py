
import os, glob, shutil, pickle
import numpy as np

from compute_dists import l2_dist, compute_dists
from optimize_param import ParamOptimizer
from optimize_lbd import LambdaOptimizer

import logging

class AlphaOptimizer(ParamOptimizer):
    par_step = 0.05
    par_min = 0.0
    par_key = lambda self,par: "%.3f" % (1.0-par)
    par_name = "alpha"
    tol = 4e-1

    def check_winner(self):
        # Check if winning directory already exists
        suffix_win = "-%s-%s" % (self.par_name, self.distname)
        prefix_win = self.dirname("")
        output_dir_win = "%s*%s" % (prefix_win, suffix_win)
        match_dir = glob.glob(output_dir_win)
        if len(match_dir) > 0:
            alpha_lbd = match_dir[0].replace(suffix_win,"").replace(prefix_win,"")
            alpha = 1-float(alpha_lbd.partition("-")[0])
            self.compute(alpha)
            self.result = alpha

    def create_winner(self):
        # create copy of winning directory
        output_dir = self.dirname(self.par_key(self.result))
        lbd_suffix = "-lbd-%s" % self.distname
        match_dir = glob.glob("%s*%s" % (output_dir, lbd_suffix))[0]
        output_dir_new = "%s-%s-%s" % \
            (match_dir.replace(lbd_suffix,""), self.par_name, self.distname)
        shutil.copytree(match_dir, output_dir_new)

    def compute(self, alpha):
        output_dir = self.dirname(self.par_key(alpha))

        if not os.path.exists(output_dir):
            shutil.copytree(self.basedir, output_dir,
                            ignore=shutil.ignore_patterns("*.log","*.zip"))

        data_file = os.path.join(output_dir, 'data.pickle')
        data = pickle.load(open(data_file, 'rb'))
        if data.bounds is None or data.bounds_alpha != 1.0-alpha:
            data.init_bounds(1.0-alpha)
            pickle.dump(data, open(data_file, 'wb'))

        exp_params = ['conf_lvl=%f' % (1.0-alpha), self.params[1]]
        if len(self.params[0]) > 0:
            exp_params[0] = '%s,%s' % (exp_params[0], self.params[0])
        logging.info("=> Optimizing lambda for dist '%s' and basedir '%s'..." % \
              (parsed_args.dist, output_dir))
        opt = LambdaOptimizer(self.experiment, self.model, basedir=output_dir,
                              dist=self.dist, resume=self.resume,
                              redist=self.redist, cvx=self.cvx,
                              params=exp_params)
        opt.run()
        lbd = opt.result
        d = opt.fulldists[opt.par_key(lbd)]
        self.fulldists[self.par_key(alpha)] = d
        d_sum = np.sum(d)
        logging.info("%s=%s: %s=%.5f (min: %.5f, max: %.5f)" % \
            (self.par_name, self.par_key(alpha),
             self.distname, d_sum, np.amin(d), np.amax(d)))
        self.dists[self.par_key(alpha)] = d_sum

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Optimize confidence level.")
    parser.add_argument('experiment', metavar='EXPERIMENT', type=str)
    parser.add_argument('model', metavar='MODEL', type=str)
    parser.add_argument('--basedir', metavar='BASENAME', type=str, default="")
    parser.add_argument('--model-params', metavar='PARAMS', type=str, default="")
    parser.add_argument('--solver-params', metavar='PARAMS', type=str, default="")
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
    logging.info("==> Optimizing alpha for dist '%s' and basedir '%s'..." % \
          (parsed_args.dist, basedir))
    opt = AlphaOptimizer(exp.Experiment, parsed_args.model, basedir=basedir,
                         dist=distfun, resume=parsed_args.resume,
                         redist=parsed_args.redist, cvx=parsed_args.cvx,
                         params=(parsed_args.model_params, parsed_args.solver_params))
    opt.run()
    logging.info("==> Optimal alpha for dist '%s' and basedir '%s': %.4f" % \
          (parsed_args.dist, basedir, 1-opt.result))
    logging.info("")
