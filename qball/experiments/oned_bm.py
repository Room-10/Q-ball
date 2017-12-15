
import logging
import numpy as np

from qball.experiments import QBallExperiment
import qball.tools.gen as gen

class MyExperiment(QBallExperiment):
    name = "1d-bm"
    pd_solver_params = {
        'n_w_tvw': { 'step_factor': 0.0005, 'step_bound': 1.2, }, # 0.993
        'sh_w_tvw': { 'step_factor': 0.001, 'step_bound': 0.08, }, # 0.0651
        'sh_l_tvc': { 'step_factor': 0.1, 'step_bound': 0.0013, }, # 0.00104
        'sh_l_tvo': { 'step_factor': 0.29, 'step_bound': 0.0014, }, # 0.00105
        'sh_l_tvw': { 'step_factor': 0.01, 'step_bound': 0.0014, }, # 0.00105
        'sh_bndl1_tvc': { 'step_factor': 0.1, 'step_bound': 0.0013, }, # 0.00105
        'sh_bndl2_tvc': { 'step_factor': 0.1, 'step_bound': 0.0013, }, # 0.00105
        'sh_bndl2_tvw': { 'step_factor': 0.1, 'step_bound': 0.0013, }, # 0.00105
    }

    def __init__(self, args):
        QBallExperiment.__init__(self, args)
        self.params['plot'].update({
            'scale': 1.0,
            'norm': False,
            'slice': (slice(None),None,None),
            'spacing': False,
        })

    def setup_imagedata(self):
        logging.info("Data setup.")
        S_data_orig, S_data, gtab = gen.synth_bimodals()
        self.data = {
            'gtab': gtab,
            'raw': S_data[:,None,None,:],
            'ground-truth': S_data_orig[:,None,None,:],
            'slice': (slice(None),0,0),
            'normed': True,
        }
