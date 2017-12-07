
import shutil, glob
import numpy as np

from compute_dists import l2_dist, load_b_sph

class ParamOptimizer(object):
    par_step = 1.0
    par_min = 0.0
    par_key = lambda self,par: "%.4f" % par
    par_name = "lbd"
    tol = 4e-2

    def __init__(self, experiment, model, basedir=None, dist=l2_dist,
                 resume=False, redist=False, cvx=False, params=""):
        self.experiment = experiment
        self.model = model
        if basedir is not None:
            self.basedir = basedir.rstrip("/")
        else:
            exp_args = [self.model, '--plot','no']
            exp = self.experiment(exp_args)
            exp.load_imagedata()
            self.basedir = exp.output_dir
        self.dist = dist
        self.distname = self.dist.__name__.replace("_dist","")
        self.resume = resume
        self.redist = redist
        self.cvx = cvx
        self.params = params

        self.result = None
        self.dists = {}
        self.fulldists = {}
        self.b_sph = load_b_sph(self.basedir)
        self.check_winner()

    def check_winner(self):
        # Check if winning directory already exists
        suffix_win = "-%s-%s" % (self.par_name, self.distname)
        prefix_win = self.dirname("")
        output_dir_win = "%s*%s" % (prefix_win, suffix_win)
        match_dir = glob.glob(output_dir_win)
        if len(match_dir) > 0:
            par = float(match_dir[0].replace(suffix_win,"").replace(prefix_win,""))
            self.compute(par)
            self.result = par

    def create_winner(self):
        # create copy of winning directory
        output_dir = self.dirname(self.par_key(self.result))
        output_dir_new = "%s-%s-%s" % (output_dir, self.par_name, self.distname)
        shutil.copytree(output_dir, output_dir_new)

    def dist_relgap(self, p1, p2):
        return (self.get_dist(p1) - self.get_dist(p2))/self.get_dist(p2)

    def get_dist(self, par):
        if self.par_key(par) not in self.dists:
            self.compute(par)
        return self.dists[self.par_key(par)]

    def run(self):
        if self.result is not None:
            return

        par_l, par, par_r = self.par_min, self.par_min, self.par_step
        relgap_l = -1.

        # increase par until dists stop increasing
        while relgap_l < 0:
            par_l = par
            par = par_r
            par_r += self.par_step
            relgap_l = self.dist_relgap(par_l, par)

        # increase par until dists stop decreasing
        while True:
            relgap_r = self.dist_relgap(par, par_r)
            if relgap_r < 0:
                break
            else:
                par_l = par
                par = par_r
                par_r += self.par_step

        # bisect
        relgap = (par_r - par_l)/par
        next = "l"
        while relgap > self.tol:
            # choose the smaller side, as long as the dist. to both is similar
            if np.abs((par_r - par) - (par - par_l)) < 1e-3:
                if self.dist_relgap(par_l, par_r) < 0:
                    next = "l"
                else:
                    next = "r"

            # bisect and reset interval
            if next == "l":
                par_new = (par_l + par)/2
                if self.dist_relgap(par_new, par) < 0:
                    par_l, par, par_r = par_l, par_new, par
                else:
                    par_l = par_new
                next = "r"
            else:
                par_new = (par + par_r)/2
                if self.dist_relgap(par_new, par) < 0:
                    par_l, par, par_r = par, par_new, par_r
                else:
                    par_r = par_new
                next = "l"
            relgap = (par_r - par_l)/par

        # set `result` variable
        self.result = par
        self.create_winner()

    def dirname(self, key):
        output_dir = self.basedir
        if self.basedir.find(self.model) < 0:
            output_dir = "%s-%s" % (output_dir, self.model)
        return "%s-%s" % (output_dir, key)

    def compute(self, par):
        # make sure to set `self.dists[self.par_key(par)]`
        pass
