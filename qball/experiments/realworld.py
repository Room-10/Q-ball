
import logging
import numpy as np

from qball.experiments import QBallExperiment
import qball.tools.gen as gen

class MyExperiment(QBallExperiment):
    name = "realworld"
    pd_solver_params = {
        'n_w_tvw': {
            'step_factor': 0.0001,
            'step_bound': 1.27, # 0.992
        },
        'sh_w_tvw': {
            'step_factor': 0.005,
            'step_bound': 0.08, # 0.0655
        },
        'sh_l_tvc': {
            'step_factor': 0.1,
            'step_bound': 0.0014, # 0.00113
        },
        'sh_l_tvo': {
            'step_factor': 0.29,
            'step_bound': 0.0014, # 0.00114
        },
        'sh_l_tvw': {
            'step_factor': 0.033,
            'step_bound': 0.0014, # 0.00114
        },
    }

    def __init__(self, args):
        QBallExperiment.__init__(self, args)
        self.params['base']['assume_normed'] = False
        self.plot_scale = 2.4
        self.plot_norm = True

    def setup_imagedata(self):
        logging.info("Data setup.")
        #np.random.seed(seed=234234)
        self.S_data_orig, self.S_data, self.gtab = gen.rw_stanford(snr=40)