
import logging
import os
import numpy as np

from dipy.viz import fvtk

from repyducible.experiment import Experiment as BaseExperiment

class Experiment(BaseExperiment):
    extra_source_files = ['demo.py','README.md']

    def init_params(self, *args):
        BaseExperiment.init_params(self, *args)
        self.params.update({
            'data': {},
            'plot': {
                'scale': 2.4,
                'norm': True,
                'spacing': True,
                'records': [],
            }
        })
        if self.pargs.solver != "cvx":
            self.params['solver']['steps'] = "precond"

    def restore_data(self, *args):
        BaseExperiment.restore_data(self, *args)
        self.data.init_spheres()
        self.data.init_odf()

    def postprocessing(self):
        l_labels = np.count_nonzero(self.data.gtab.bvals)
        imagedims = self.data.raw[self.data.slice].shape[:-1]
        resultx = self.model.x.vars(self.result['data'][0], True)
        try:
            resultu = resultx['u']
        except:
            resultu = resultx['u1']
        self.upd = resultu.reshape(imagedims + (l_labels,))

    def plot(self):
        if self.pargs.plot == "no":
            return

        p = self.params['plot']

        from qball.tools.plot import plot_as_odf, prepare_odfplot
        if self.pargs.plot == "show":
            logging.info("Plotting results...")
            data = [self.data.odf_ground_truth, self.data.odf, self.upd]
            r = prepare_odfplot(data, p, self.data.dipy_sph)
            fvtk.show(r, size=(1024, 768))

        logging.info("Recording plot...")
        p['records'] = [p['slice']] if len(p['records']) == 0 else p['records']
        imgdata = [
            (self.upd, "upd"),
            (self.data.odf, "fin"),
            (self.data.odf_ground_truth, "fin_orig")
        ]
        plot_as_odf(imgdata, p, self.data.dipy_sph, self.output_dir)

        if hasattr(self.data, "phantom"):
            phantom_plot_file = os.path.join(self.output_dir, "plot-phantom.pdf")
            self.data.phantom.plot_phantom(output_file=phantom_plot_file)

    def plot_dti(self):
        from qball.tools.plot import plot_as_dti
        plot_as_dti(self.data.gtab, self.data.raw[self.data.slice],
                    self.params['plot'], self.data.dipy_sph, self.output_dir)
