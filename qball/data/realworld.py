
from qball.data import QBallData
import qball.tools.gen as gen

class Data(QBallData):
    name = "realworld"
    default_params = {
        'model': {},
        'solver': {
            'pdhg': {
                'n_w_tvw': { 'step_factor': 0.0001, 'step_bound': 1.27, },
                'sh_w_tvw': { 'step_factor': 0.005, 'step_bound': 0.08, },
                'sh_l_tvc': { 'step_factor': 0.1, 'step_bound': 0.0014, },
                'sh_l_tvo': { 'step_factor': 0.29, 'step_bound': 0.0014, },
                'sh_l_tvw': { 'step_factor': 0.033, 'step_bound': 0.0014, },
                'sh_bndl1_tvc': { 'step_factor': 0.1, 'step_bound': 0.0014, },
                'sh_bndl2_tvc': { 'step_factor': 0.1, 'step_bound': 0.0014, },
                'sh_bndl2_tvw': { 'step_factor': 0.033, 'step_bound': 0.0014, },
            },
        },
        'plot': {
            'scale': 1.8,
            'slice': (slice(None),slice(None),None),
        },
    }

    def __init__(self, *args):
        QBallData.__init__(self, *args)
        self.ground_truth, self.raw, self.gtab = gen.rw_stanford(snr=None)
        self.slice = (slice(20,50), slice(55,85), 38)
        self.normed = False
        """
        # from dipy qbi-csa example (used for SSVM):
        # http://nipy.org/dipy/examples_built/reconst_csa.html
        maskdata, mask = median_otsu(self.raw, median_radius=3, numpass=1,
            autocrop=True, vol_idx=range(10, 50), dilate=2)
        self.raw = np.array(maskdata[13:43, 44:74, 28], order='C')
        """
