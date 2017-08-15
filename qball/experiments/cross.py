
import os, logging
import numpy as np

from qball.experiments import QBallExperiment
import qball.tools.gen as gen

class MyExperiment(QBallExperiment):
    name = "cross"
    pd_solver_params = {
        'n_w_tvw': {
            'step_factor': 0.0001,
            'step_bound': 1.25, # 0.971
        },
        'sh_w_tvw': {
            'step_factor': 0.001,
            'step_bound': 0.08, # 0.0654
        },
        'sh_l_tvc': {
            'step_factor': 0.1,
            'step_bound': 0.00135, # 0.00105
        },
        'sh_l_tvo': {
            'step_factor': 0.29,
            'step_bound': 0.0014, # 0.00105
        },
        'sh_l_tvw': {
            'step_factor': 0.033,
            'step_bound': 0.0014, # 0.00105
        },
    }

    def setup_imagedata(self):
        logging.info("Data setup.")
        #np.random.seed(seed=234234)
        self.S_data_orig, self.S_data, \
            self.gtab, self.phantom = gen.synth_cross(snr=20)

    def plot(self):
        QBallExperiment.plot(self)
        if hasattr(self, "phantom"):
            phantom_plot_file = os.path.join(self.output_dir, "plot-phantom.pdf")
            self.phantom.plot_phantom(output_file=phantom_plot_file)
