
import logging
import numpy as np

from qball.experiments import QBallExperiment
import qball.tools.gen as gen

class MyExperiment(QBallExperiment):
    name = "realworld"
    pd_solver_params = {
        'n_w_tvw': { 'step_factor': 0.0001, 'step_bound': 1.27, }, # 0.992
        'sh_w_tvw': { 'step_factor': 0.005, 'step_bound': 0.08, }, # 0.0655
        'sh_l_tvc': { 'step_factor': 0.1, 'step_bound': 0.0014, }, # 0.00113
        'sh_l_tvo': { 'step_factor': 0.29, 'step_bound': 0.0014, }, # 0.00114
        'sh_l_tvw': { 'step_factor': 0.033, 'step_bound': 0.0014, }, # 0.00114
        'sh_bndl1_tvc': { 'step_factor': 0.1, 'step_bound': 0.0014, }, # 0.00105
        'sh_bndl2_tvc': { 'step_factor': 0.1, 'step_bound': 0.0014, }, # 0.00105
        'sh_bndl2_tvw': { 'step_factor': 0.033, 'step_bound': 0.0014, }, # 0.00105
    }

    def __init__(self, args):
        QBallExperiment.__init__(self, args)
        self.params['base']['assume_normed'] = False
        self.params['plot'].update({
            'scale': 1.8,
            'slice': (slice(None),slice(None),None),
        })

    def setup_imagedata(self):
        logging.info("Data setup.")
        S_data_orig, S_data, gtab, resp = gen.rw_stanford(snr=None, csd=True)
        self.data = {
            'gtab': gtab,
            'raw': S_data,
            'ground-truth': S_data_orig,
            'slice': (slice(20,50), slice(55,85), 38),
            'normed': False,
        }
        """
        # from dipy qbi-csa example (used for SSVM):
        # http://nipy.org/dipy/examples_built/reconst_csa.html
        maskdata, mask = median_otsu(data, median_radius=3, numpass=1,
            autocrop=True, vol_idx=range(10, 50), dilate=2)
        S_data = np.array(maskdata[13:43, 44:74, 28], order='C')
        """

        if self.model_name in ["sh_w_tvw", "n_w_tvw"]:
            self.data['csd_response'] = resp
