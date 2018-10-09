
from qball.data import QBallData
import qball.tools.gen as gen

class Data(QBallData):
    name = "1d"
    default_params = {
        'model': {},
        'solver': {
            'pdhg': {
                'n_w_tvw': { 'step_factor': 0.0005, 'step_bound': 1.2, },
                'sh_w_tvw': { 'step_factor': 0.001, 'step_bound': 0.08, },
                'sh_l_tvc': { 'step_factor': 0.1, 'step_bound': 0.0013, },
                'sh_l_tvo': { 'step_factor': 0.29, 'step_bound': 0.0014, },
                'sh_l_tvw': { 'step_factor': 0.01, 'step_bound': 0.0014, },
                'sh_bndl1_tvc': { 'step_factor': 0.1, 'step_bound': 0.0013, },
                'sh_bndl2_tvc': { 'step_factor': 0.1, 'step_bound': 0.0013, },
                'sh_bndl2_tvw': { 'step_factor': 0.1, 'step_bound': 0.0013, },
            },
        },
        'plot': {
            'scale': 1.0,
            'norm': False,
            'slice': (slice(None),None,None),
            'spacing': False,
        },
    }

    def __init__(self, *args):
        QBallData.__init__(self, *args)
        S_data_orig, S_data, self.gtab = gen.synth_unimodals()
        self.raw = S_data[:,None,None,:]
        self.ground_truth = S_data_orig[:,None,None,:]
        self.slice = (slice(None),0,0)
        self.normed = True
