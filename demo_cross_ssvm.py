
"""
    This standalone application applies several reconstruction themes to a 1d
    Q-ball data set, i.e. the input image is an ODF in each pixel.
"""

from __future__ import division

import os, logging, sys
import numpy as np

import util
import tools_gen as gen
from models import SSVMModel

class MyExperiment(util.QBallExperiment):
    Model = SSVMModel

    def __init__(self, args):
        util.QBallExperiment.__init__(self, "cross-ssvm", args)
        self.params['fit'] = {
            'sphere': None,
            'solver_params': {
                'lbd': 2.5,
                'term_relgap': 1e-05,
                'term_maxiter': 100000,
                'granularity': 5000,
                'step_factor': 0.0001,
                'step_bound': 1.3,
                'dataterm': "W1",
                'use_gpu': True
            },
        }

    def solve(self):
        self.params['fit']['sphere'] = self.qball_sphere
        util.QBallExperiment.solve(self)

    def setup_imagedata(self):
        logging.info("Data setup.")
        #np.random.seed(seed=234234)
        self.S_data_orig, self.S_data, \
            self.gtab, self.phantom = gen.synth_cross(snr=20)

    def plot(self):
        util.QBallExperiment.plot(self)
        phantom_plot_file = os.path.join(self.output_dir, "plot-phantom.pdf")
        self.phantom.plot_phantom(output_file=phantom_plot_file)

if __name__ == "__main__":
    logging.info("Running from command line: %s" % sys.argv)
    exp = MyExperiment(sys.argv[1:])
    exp.run()
    exp.plot()
