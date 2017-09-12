
import numpy as np

import sys, os, pickle
from argparse import ArgumentParser

import dipy.core.sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.viz import fvtk

import logging

from qball.util import add_log_file, output_dir_name, output_dir_create, \
                       data_from_file, backup_source

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
        parser.add_argument('--params', metavar='SOLVER_PARAMS',
                            default='', type=str,
                            help="Params to be passed to the solver.")
        parsed_args = parser.parse_args(args)
        self.model_name = parsed_args.model
        if parsed_args.output == '':
            self.output_dir = output_dir_name("%s-%s" % (self.name, self.model_name))
        else:
            self.output_dir = parsed_args.output
        self.user_params = eval("dict(%s)" % parsed_args.params)
        self.cvx = parsed_args.cvx
        self.resume = parsed_args.resume
        self.plot_mode = parsed_args.plot

        if parsed_args.seed is not None:
            np.random.seed(seed=parsed_args.seed)

        output_dir_create(self.output_dir)
        add_log_file(logging.getLogger(), self.output_dir)
        backup_source(self.output_dir)
        logging.info("Args: %s" % args)

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

    def load_imagedata(self): pass

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
        self.params['fit']['solver_params'].update(self.user_params)

    def load_imagedata(self):
        gtab_file = os.path.join(self.output_dir, 'gtab.pickle')
        S_data_file = os.path.join(self.output_dir, 'S_data.np')
        S_data_orig_file = os.path.join(self.output_dir, 'S_data_orig.np')

        self.gtab = data_from_file(gtab_file, format="pickle")
        self.S_data = data_from_file(S_data_file)
        self.S_data_orig = data_from_file(S_data_orig_file)
        if self.S_data is None:
            self.setup_imagedata()
            pickle.dump(self.gtab, open(gtab_file, 'wb'))
            np.save(open(S_data_file, 'wb'), self.S_data)
            np.save(open(S_data_orig_file, 'wb'), self.S_data_orig)
        self.imagedims = self.S_data.shape[:-1]
        b_vecs = self.gtab.bvecs[self.gtab.bvals > 0,...]
        self.qball_sphere = dipy.core.sphere.Sphere(xyz=b_vecs)

    def solve(self):
        if self.model_name == 'n_w_tvw':
            self.params['fit']['sphere'] = self.qball_sphere

        if self.resume and self.pd_result is not None:
            self.continue_at = self.pd_result
            self.params['fit']['solver_params']['continue_at'] = self.continue_at

        import qball.models
        MyModel = getattr(qball.models, '%s_Model' % self.model_name)
        self.model = MyModel(self.gtab, **self.params['base'])
        self.model.fit(self.S_data, **self.params['fit'])
        self.pd_result = self.model.solver_state
        self.details = self.model.solver_details

    def postprocessing(self):
        l_labels = self.upd.shape[0]
        imagedims = self.upd.shape[1:]
        self.upd = self.upd.reshape(l_labels, -1)
        self.upd = np.array(self.upd.T, order='C').reshape(imagedims + (l_labels,))

        basemodel = CsaOdfModel(self.gtab, **self.params['base'])
        f = basemodel.fit(self.S_data).odf(self.qball_sphere)
        self.fin = np.clip(f, 0, np.max(f, -1)[..., None])
        f = basemodel.fit(self.S_data_orig).odf(self.qball_sphere)
        self.fin_orig = np.clip(f, 0, np.max(f, -1)[..., None])

    def prepare_plot(self, data, p_slice=None):
        p = self.params['plot']
        p_slice = p['slice'] if p_slice is None else p_slice
        data = (data,) if type(data) is np.ndarray else data

        slicedims = data[0][p_slice].shape[:-1]
        l_labels = data[0].shape[-1]

        axes = [0,1,2]
        long_ax = np.argmax(slicedims)
        axes.remove(long_ax)
        stack_ax = axes[0]
        if slicedims[axes[0]] < slicedims[axes[1]]:
            stack_ax = axes[1]
        axes.remove(stack_ax)
        view_ax = axes[0]

        camera_params = {
            'position': [0,0,0],
            'view_up': [0,0,0]
        }
        camera_params['view_up'][max(long_ax,stack_ax)] = 1
        dist = 2*p['scale']*max(slicedims[long_ax],len(data)*(slicedims[stack_ax]+1)-1)
        camera_params['position'][view_ax] = -dist if view_ax == 1 else dist

        stack = [u[p_slice] for u in data]
        if p['spacing']:
            uniform_odf = np.ones((1,1,1,l_labels), order='C')/(4*np.pi)
            tile_descr = [1,1,1,1]
            tile_descr[long_ax] = slicedims[long_ax]
            spacing = np.tile(uniform_odf, tile_descr)
            for i in reversed(range(1,len(stack))):
                stack.insert(i, spacing)
        if stack_ax == max(long_ax,stack_ax):
            stack = list(reversed(stack))

        plotdata = np.concatenate(stack, axis=stack_ax)
        r = fvtk.ren()
        r_data = fvtk.sphere_funcs(plotdata, self.qball_sphere, colormap='jet',
                                   norm=p['norm'], scale=p['scale'])
        fvtk.add(r, r_data)
        r.set_camera(**camera_params)
        return r

    def plot(self):
        if self.plot_mode == "no":
            return

        p = self.params['plot']

        if self.plot_mode == "show":
            logging.info("Plotting results...")
            r = self.prepare_plot([self.fin_orig, self.fin, self.upd])
            fvtk.show(r, size=(1024, 768))

        logging.info("Recording plot...")
        p['records'] = [p['slice']] if len(p['records']) == 0 else p['records']
        imgdata = [
            (self.upd, "upd"),
            (self.fin, "fin"),
            (self.fin_orig, "fin_orig")
        ]
        for img, name in imgdata:
            for i,s in enumerate(p['records']):
                fname = os.path.join(self.output_dir, "plot-%s-%d.png" % (name,i))
                r = self.prepare_plot(img, p_slice=s)
                fvtk.snapshot(r, size=(1500,1500), offscreen=True, fname=fname)
