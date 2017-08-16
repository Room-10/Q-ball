
import logging
import numpy as np

from qball.experiments import QBallExperiment
import qball.tools.gen as gen

class MyExperiment(QBallExperiment):
    name = "1d"
    pd_solver_params = {
        'n_w_tvw': {
            'step_factor': 0.0005,
            'step_bound': 1.2, # 0.993
        },
        'sh_w_tvw': {
            'step_factor': 0.001,
            'step_bound': 0.08, # 0.0651
        },
        'sh_l_tvc': {
            'step_factor': 0.1,
            'step_bound': 0.0013, # 0.00104
        },
        'sh_l_tvo': {
            'step_factor': 0.29,
            'step_bound': 0.0014, # 0.00105
        },
        'sh_l_tvw': {
            'step_factor': 0.01,
            'step_bound': 0.0014, # 0.00105
        },
    }

    def setup_imagedata(self):
        logging.info("Data setup.")
        self.S_data_orig, self.S_data, self.gtab = gen.synth_unimodals()
