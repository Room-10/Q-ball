
import numpy as np

import sys, os, pickle
from argparse import ArgumentParser

import dipy.core.sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.viz import fvtk

import logging

from qball.util import add_log_file, output_dir_name, output_dir_create, \
                       data_from_file, backup_source
from qball.sphere import load_sphere

class Experiment(object):
    """ Base class for experiments. """

    def __init__(self, name, args):
        self.name = name

        parser = ArgumentParser(description="See README.md.")
        parser.add_argument('model', metavar='MODEL_NAME', type=str,
                            help="Name of model to use.")
        parser.add_argument('--output', metavar='OUTPUT_DIR',
                            default='', type=str,
                            help="Path to output directory. "
                                   + "Existing data will be loaded and used.")
        parser.add_argument('--resume', action="store_true", default=False,
                            help="Continue at last state.")
        parser.add_argument('--plot', metavar='PLOT_MODE', default="show",
                            type=str, help="Plot mode (show|hide|no).")
        parser.add_argument('--cvx', action="store_true", default=False,
                            help="Use CVX as solver engine.")
        parser.add_argument('--seed', metavar="SEED", default=None, type=int,
                            help="Random seed for noise generation.")
        parser.add_argument('--model-params', metavar='PARAMS',
                            default='', type=str,
                            help="Params to be applied to the model.")
        parser.add_argument('--solver-params', metavar='PARAMS',
                            default='', type=str,
                            help="Params to be passed to the solver.")
        parsed_args = parser.parse_args(args)
        self.model_name = parsed_args.model
        if parsed_args.output == '':
            self.output_dir = output_dir_name("%s-%s" % (self.name, self.model_name))
        else:
            self.output_dir = parsed_args.output
        self.user_params = (eval("dict(%s)" % parsed_args.model_params),
                            eval("dict(%s)" % parsed_args.solver_params))
        self.cvx = parsed_args.cvx
        self.resume = parsed_args.resume
        self.plot_mode = parsed_args.plot

        if parsed_args.seed is not None:
            np.random.seed(seed=parsed_args.seed)

        output_dir_create(self.output_dir)
        add_log_file(logging.getLogger(), self.output_dir)
        backup_source(self.output_dir)
        logging.debug("Args: %s" % args)

        self.init_params()
        self.restore_from_output_dir()
        self.apply_user_params()

    def init_params(self):
        self.params = {}

    def apply_user_params(self): pass

    def restore_from_output_dir(self):
        self.load_imagedata()

        self.params_file = os.path.join(self.output_dir, 'params.pickle')
        self.pd_result_file = os.path.join(self.output_dir, 'result_raw.pickle')
        self.details_file = os.path.join(self.output_dir, 'details.pickle')

        self.pd_result = data_from_file(self.pd_result_file, format="pickle")
        if self.pd_result is not None:
            self.details = data_from_file(self.details_file, format="pickle")

        params = data_from_file(self.params_file, format="pickle")
        if params is not None:
            self.params.update(params)

    def setup_imagedata(self): pass

    def load_imagedata(self):
        data_file = os.path.join(self.output_dir, 'data.pickle')
        self.data = data_from_file(data_file, format="pickle")
        if self.data is None:
            self.setup_imagedata()
            pickle.dump(self.data, open(data_file, 'wb'))

    def postprocessing(self): pass

    def solve(self): pass

    def run(self):
        pickle.dump(self.params, open(self.params_file, 'wb'))
        if self.pd_result is None or self.resume:
            self.solve()
            pickle.dump(self.pd_result, open(self.pd_result_file, 'wb'))
            pickle.dump(self.details, open(self.details_file, 'wb'))

        try:
            self.upd = self.pd_result[0]['u'].copy()
        except KeyError:
            self.upd = self.pd_result[0]['u1'].copy()

        self.postprocessing()

    def plot(self): pass

class QBallExperiment(Experiment):
    pd_solver_params = None
    name = ""

    def __init__(self, args):
        Experiment.__init__(self, self.name, args)

    def init_params(self):
        Experiment.init_params(self)
        self.params['model'] = self.model_name
        self.params['fit'] = {
            'model_params': {},
            'solver_engine': 'cvx' if self.cvx else 'pd',
            'solver_params': {},
        }
        if not self.cvx:
            self.params['fit']['solver_params'].update(
                self.pd_solver_params[self.model_name]
            )
        self.params['base'] = {
            'sh_order': 6,
            'assume_normed': True
        }
        self.params['plot'] = {
            'scale': 2.4,
            'norm': True,
            'spacing': True,
            'records': [],
        }

    def apply_user_params(self):
        self.params['fit']['model_params'].update(self.user_params[0])
        self.params['fit']['solver_params'].update(self.user_params[1])

        if "bnd" in self.model_name:
            alpha = self.params['fit']['model_params'].get('conf_lvl', 0.9)
            if 'bounds' not in self.data or self.data['bounds'][0] != alpha:
                from qball.tools.bounds import compute_hardi_bounds
                compute_hardi_bounds(self.data, alpha=alpha)
                data_file = os.path.join(self.output_dir, 'data.pickle')
                pickle.dump(self.data, open(data_file, 'wb'))
            self.params['fit']['model_params']['conf_lvl'] = alpha

    def load_imagedata(self):
        Experiment.load_imagedata(self)
        self.params['base']['assume_normed'] = self.data['normed']
        gtab = self.data['gtab']
        b_vecs = gtab.bvecs[gtab.bvals > 0,...]
        self.data['b_sph'] = load_sphere(vecs=b_vecs.T)
        self.qball_sphere = dipy.core.sphere.Sphere(xyz=b_vecs)

    def solve(self):
        if self.model_name == 'n_w_tvw':
            self.params['fit']['model_params']['sphere'] = self.qball_sphere

        if self.resume and self.pd_result is not None:
            self.continue_at = self.pd_result
            self.params['fit']['solver_params']['continue_at'] = self.continue_at

        import qball.models
        MyModel = getattr(qball.models, '%s_Model' % self.model_name)
        self.model = MyModel(self.data['gtab'], **self.params['base'])
        self.model.fit(self.data, **self.params['fit'])
        self.pd_result = self.model.solver_state
        self.details = self.model.solver_details

    def postprocessing(self):
        if 'csd_response' in self.data \
           and self.data['csd_response'] is not None:
            logging.info("Using CSD for ground truth reconstruction.")
            basemodel = ConstrainedSphericalDeconvModel(self.data['gtab'], \
                self.data['csd_response'])
        else:
            basemodel = CsaOdfModel(self.data['gtab'], **self.params['base'])

        S_data = self.data['raw'][self.data['slice']]
        S_data_orig = self.data['ground-truth'][self.data['slice']]
        f = basemodel.fit(S_data).odf(self.qball_sphere)
        self.fin = np.clip(f, 0, np.max(f, -1)[..., None])
        f = basemodel.fit(S_data_orig).odf(self.qball_sphere)
        self.fin_orig = np.clip(f, 0, np.max(f, -1)[..., None])

        l_labels = np.count_nonzero(self.data['gtab'].bvals)
        imagedims = S_data.shape[:-1]
        self.upd = self.upd.reshape(imagedims + (l_labels,))

    def plot(self):
        if self.plot_mode == "no":
            return

        p = self.params['plot']

        from qball.tools.plot import plot_as_odf, prepare_odfplot
        if self.plot_mode == "show":
            logging.info("Plotting results...")
            data = [self.fin_orig, self.fin, self.upd]
            r = prepare_odfplot(data, p, self.qball_sphere)
            fvtk.show(r, size=(1024, 768))

        logging.info("Recording plot...")
        p['records'] = [p['slice']] if len(p['records']) == 0 else p['records']
        imgdata = [
            (self.upd, "upd"),
            (self.fin, "fin"),
            (self.fin_orig, "fin_orig")
        ]
        plot_as_odf(imgdata, p, self.qball_sphere, self.output_dir)

    def plot_dti(self):
        from qball.tools.plot import plot_as_dti
        plot_as_dti(self.data['gtab'], self.data['raw'][self.data['slice']],
                    self.params['plot'], self.qball_sphere, self.output_dir)

