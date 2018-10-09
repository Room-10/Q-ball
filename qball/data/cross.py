
from qball.data import QBallData
import qball.tools.gen as gen

class Data(QBallData):
    name = "cross"
    default_params = {
        'model': {},
        'solver': {
            'pdhg': {
                'n_w_tvw': { 'step_factor': 0.0001, 'step_bound': 1.25, },
                'sh_w_tvw': { 'step_factor': 0.001, 'step_bound': 0.08, },
                'sh_l_tvc': { 'step_factor': 0.1, 'step_bound': 0.00135, },
                'sh_l_tvo': { 'step_factor': 0.29, 'step_bound': 0.0014, },
                'sh_l_tvw': { 'step_factor': 0.033, 'step_bound': 0.0014, },
                'sh_l_tvw2': { 'step_factor': 0.01, 'step_bound': 0.00135, },
                'sh_bndl1_tvc': { 'step_factor': 0.1, 'step_bound': 0.00139, },
                'sh_bndl2_tvc': { 'step_factor': 0.1, 'step_bound': 0.00139, },
                'sh_bndl2_tvw': { 'step_factor': 0.033, 'step_bound': 0.00139, },
            },
        },
        'plot': {
            'scale': 0.5,
            'norm': False,
            'slice': (slice(None),slice(None),None),
        },
    }

    def __init__(self, *args):
        QBallData.__init__(self, *args)
        S_data_orig, S_data, self.gtab, self.phantom = gen.synth_cross()
        self.raw = S_data[:,:,None,:]
        self.ground_truth = S_data_orig[:,:,None,:]
        self.slice = (slice(None),slice(None),0)
        self.normed = True
