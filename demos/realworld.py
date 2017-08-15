
"""
    This standalone application applies several reconstruction themes to a
    HARDI data set.
"""

from __future__ import division

import os, logging, sys
import numpy as np

try:
    import qball
except:
    import set_qball_path
import qball.util as util
import qball.tools.gen as gen

fit_params = {
    'n_w_tvw': {
        'sphere': None,
        'solver_params': {
            'lbd': 1.5,
            'term_relgap': 1e-05,
            'term_maxiter': int(1e7),
            'granularity': 10000,
            'step_factor': 0.0001,
            'step_bound': 1.27, # 0.992
            'dataterm': "W1",
            'use_gpu': True
        },
    },
    'sh_w_tvw': {
        'solver_engine': 'pd',
        'solver_params': {
            'lbd': 0.8,
            'term_relgap': 1e-05,
            'term_maxiter': int(1e7),
            'granularity': 10000,
            'step_factor': 0.005,
            'step_bound': 0.08, # 0.0655
            'dataterm': "W1",
            'use_gpu': True
        },
    },
    'sh_l_tvc': {
        'solver_engine': 'pd',
        'solver_params': {
            'lbd': 1.0,
            'term_relgap': 1e-05,
            'term_maxiter': int(1e7),
            'granularity': 10000,
            'step_factor': 0.1,
            'step_bound': 0.0014, # 0.00113
            'use_gpu': True
        },
    },
    'sh_l_tvo': {
        'solver_engine': 'pd',
        'solver_params': {
            'lbd': 1.0,
            'term_relgap': 1e-05,
            'term_maxiter': int(1e7),
            'granularity': 10000,
            'step_factor': 0.29,
            'step_bound': 0.0014, # 0.00114
            'use_gpu': True
        },
    },
    'sh_l_tvw': {
        'solver_engine': 'pd',
        'solver_params': {
            'lbd': 1.0,
            'term_relgap': 1e-05,
            'term_maxiter': int(1e7),
            'granularity': 10000,
            'step_factor': 0.033,
            'step_bound': 0.0014, # 0.00114
            'use_gpu': True
        },
    },
}

class MyExperiment(util.QBallExperiment):
    def __init__(self, args):
        util.QBallExperiment.__init__(self, "realworld", args)
        self.params['base']['assume_normed'] = False
        self.plot_scale = 2.4
        self.plot_norm = True
        if not self.cvx:
            self.params['fit'] = fit_params[self.model_name]

    def setup_imagedata(self):
        logging.info("Data setup.")
        #np.random.seed(seed=234234)
        self.S_data_orig, self.S_data, self.gtab = gen.rw_stanford(snr=40)

if __name__ == "__main__":
    logging.info("Running from command line: %s" % sys.argv)
    exp = MyExperiment(sys.argv[1:])
    exp.run()
    exp.plot()
