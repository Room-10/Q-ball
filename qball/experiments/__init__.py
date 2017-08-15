
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
        parser.add_argument('--batch', action="store_true", default=False,
                            help="Activate batch processing (no plot).")
        parser.add_argument('--cvx', action="store_true", default=False,
                            help="Use CVX as solver engine.")
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
        self.interactive = not parsed_args.batch

        output_dir_create(self.output_dir)
        add_log_file(logging.getLogger(), self.output_dir)
        backup_source(self.output_dir)
        self.params = {}

    def setup_imagedata(self): pass

    def load_imagedata(self): pass

    def postprocessing(self): pass

    def solve(self): pass

    def run(self):
        # try to load as much data from files as possible
        self.load_imagedata()

        pd_result_file = os.path.join(self.output_dir, 'result_raw.pickle')
        details_file = os.path.join(self.output_dir, 'details.pickle')
        params_file = os.path.join(self.output_dir, 'params.pickle')

        self.pd_result = data_from_file(pd_result_file, format="pickle")
        if self.pd_result is not None:
            self.details = data_from_file(details_file, format="pickle")

        params = data_from_file(params_file, format="pickle")
        if params is None:
            pickle.dump(self.params, open(params_file, 'wb'))
        else:
            self.params = params

        if self.pd_result is None or self.resume:
            self.solve()
            pickle.dump(self.pd_result, open(pd_result_file, 'wb'))
            pickle.dump(self.details, open(details_file, 'wb'))

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
        import qball.models
        self.Model = getattr(qball.models, '%s_Model' % self.model_name)
        self.params['model'] = self.model_name
        self.params['fit'] = {
            'solver_engine': 'cvx' if self.cvx else 'pd',
            'solver_params': {},
        }
        if not self.cvx:
            self.params['fit']['solver_params'].update(
                self.pd_solver_params[self.model_name]
            )
        self.params['fit']['solver_params'].update(self.user_params)
        self.params['base'] = {
            'sh_order': 6,
            'assume_normed': True
        }
        self.plot_scale = 1.0
        self.plot_norm = False

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

        self.model = self.Model(self.gtab, **self.params['base'])
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

    def plot(self):
        logging.info("Plotting results...")
        n_image = np.prod(self.imagedims)
        d_image = len(self.imagedims)
        l_labels = self.upd.shape[-1]

        # set up data to plot, including spacing in the 2d case
        stack = []
        if d_image == 2:
            uniform_odf = np.ones((l_labels,), order='C')/l_labels
            spacing = np.tile(uniform_odf, (self.imagedims[1], 1, 1, 1))
            for i, u in enumerate([self.upd, self.fin, self.fin_orig]):
                stack.append(u[:,:,None,:])
                if i < 2:
                    stack.append(spacing)
        else:
            stack = [u[:,None,None,:] for u in (self.upd, self.fin, self.fin_orig)]
        plotdata = np.concatenate(stack, axis=1)

        if self.interactive:
            # plot self.upd and self.fin as q-ball data sets
            r = fvtk.ren()
            r_data = fvtk.sphere_funcs(plotdata, self.qball_sphere, colormap='jet',
                                       norm=self.plot_norm, scale=self.plot_scale)
            fvtk.add(r, r_data)
            fvtk.show(r, size=(1024, 768))

        logging.info("Recording plot...")
        imgdata = [
            (self.upd, "upd"),
            (self.fin, "fin"),
            (self.fin_orig, "fin_orig")
        ]
        for img,name in imgdata:
            plotdata2 = img.copy()
            plotdata2.shape = self.imagedims + (1,)*(3-d_image) + (l_labels,)
            r = fvtk.ren()
            r_data = fvtk.sphere_funcs(plotdata2, self.qball_sphere, colormap='jet',
                                       norm=self.plot_norm, scale=self.plot_scale)
            fvtk.add(r, r_data)
            r.reset_clipping_range()
            fvtk.snapshot(r, size=(1500,1500), offscreen=True,
                          fname=os.path.join(self.output_dir, "plot-"+name+".png"))
