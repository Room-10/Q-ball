
import logging
import numpy as np

from qball.experiments import QBallExperiment
import qball.tools.gen as gen

class MyExperiment(QBallExperiment):
    name = "challenge"
    pd_solver_params = {
        'n_w_tvw': {
            'step_factor': 0.0001,
            'step_bound': 1.27, # 0.990
        },
        'sh_w_tvw': {
            'step_factor': 0.005,
            'step_bound': 0.115, # 0.0870
        },
        'sh_l_tvc': {
            'step_factor': 0.1,
            'step_bound': 0.0018, # 0.00137
        },
        'sh_l_tvo': {
            'step_factor': 0.29,
            'step_bound': 0.0018, # 0.00137
        },
        'sh_l_tvw': {
            'step_factor': 0.033,
            'step_bound': 0.0018, # 0.00137
        },
        'sh_bndl1_tvc': {
            'step_factor': 0.1,
            'step_bound': 0.0018, # 0.00137
        },
        'sh_bndl2_tvc': {
            'step_factor': 0.1,
            'step_bound': 0.0018, # 0.00137
        },
        'sh_bndl2_tvw': {
            'step_factor': 0.033,
            'step_bound': 0.0018, # 0.00137
        },
    }

    def __init__(self, args):
        QBallExperiment.__init__(self, args)
        self.params['base']['assume_normed'] = False
        self.params['plot'].update({
            'slice': (slice(None),None,slice(None)),
        })

    def setup_imagedata(self):
        logging.info("Data setup.")
        self.S_data_orig, self.S_data, self.gtab = gen.synth_isbi2013()
