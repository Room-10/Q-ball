
from qball.data import QBallData
import qball.tools.gen as gen

class Data(QBallData):
    name = "challenge"
    default_params = {
        'model': {},
        'solver': {
            'pdhg': {
                'n_w_tvw': { 'step_factor': 0.0001, 'step_bound': 1.27, },
                'sh_w_tvw': { 'step_factor': 0.005, 'step_bound': 0.115, },
                'sh_l_tvc': { 'step_factor': 0.1, 'step_bound': 0.0018, },
                'sh_l_tvo': { 'step_factor': 0.29, 'step_bound': 0.0018, },
                'sh_l_tvw': { 'step_factor': 0.033, 'step_bound': 0.0018, },
                'sh_bndl1_tvc': { 'step_factor': 0.1, 'step_bound': 0.0018, },
                'sh_bndl2_tvc': { 'step_factor': 0.1, 'step_bound': 0.0018, },
                'sh_bndl2_tvw': { 'step_factor': 0.033, 'step_bound': 0.0018, },
            },
        },
        'plot': {
            'slice': (slice(None),None,slice(None)),
        },
    }

    def __init__(self, *args):
        QBallData.__init__(self, *args)
        self.ground_truth, self.raw, self.gtab = gen.synth_isbi2013()
        self.slice = (slice(12,27),22,slice(21,36))
        self.normed = False
