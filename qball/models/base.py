
from qball.tools import clip_hardi_data
from qball.tools.diff import staggered_diff_avgskips

import numpy as np

import logging

from opymize.solvers.pdhg import PDHG

class BaseModel(object):
    "Base class for models that are solved using PDHG."
    def __init__(self):
        self.x = None
        self.y = None
        self.F = None
        self.G = None
        self.linop = None
        self.constvars = {}
        self.extravars = {}

    def solve(self, **solver_params):
        self.solver_params = solver_params
        if self.solver_params.get('continue_at') is None:
            solver_params['continue_at'] = (self.x.data, self.y.data)
        solver = PDHG(self.G, self.F, self.linop)
        details = solver.solve(**solver_params)
        result = solver.state
        self.x.data[:] = result[0]
        self.y.data[:] = result[1]
        return details

    @property
    def state(self):
        return (self.x, self.y)

    def run_tests(self):
        from opymize.tools.tests import test_gpu_op
        for op in [self.linop, self.linop.adjoint,
                   self.G.prox(0.5 + np.random.rand()),
                   self.F.conj.prox(0.5 + np.random.rand())]:
            test_gpu_op(op)

class ModelHARDI(BaseModel):
    "Base class for HARDI reconstruction models."
    def __init__(self, data, model_params):
        BaseModel.__init__(self)
        self.model_params = model_params
        self.data = data

        gtab = self.data['gtab']
        b_sph = self.data['b_sph']

        data = self.data['raw'][self.data['slice']]
        data = np.array(data, dtype=np.float64, copy=True)
        if not self.data['normed']:
            data.clip(1.0, out=data)
            b0 = data[...,(gtab.bvals == 0)].mean(-1)
            data /= b0[...,None]
        data = data[...,gtab.bvals > 0]

        imagedims = data.shape[:-1]
        n_image = np.prod(imagedims)
        d_image = len(imagedims)
        l_labels = b_sph.mdims['l_labels']
        s_manifold = b_sph.mdims['s_manifold']
        m_gradients = b_sph.mdims['m_gradients']
        r_points = b_sph.mdims['r_points']
        assert(data.shape[-1] == l_labels)

        self.extravars['b_sph'] = b_sph

        c = self.constvars

        c['avgskips'] = staggered_diff_avgskips(imagedims)
        c['lbd'] = self.model_params.get('lbd', 1.0)
        c['b'] = b_sph.b
        c['A'] = b_sph.A
        c['B'] = b_sph.B
        c['P'] = b_sph.P
        c['b_precond'] = b_sph.b_precond
        c['imagedims'] = imagedims
        c['l_labels'] = l_labels
        c['n_image'] = n_image
        c['m_gradients'] = m_gradients
        c['s_manifold'] = s_manifold
        c['d_image'] = d_image
        c['r_points'] = r_points

        inpaintloc = self.model_params.get('inpaintloc', np.zeros(imagedims))
        c['inpaint_nloc'] = np.ascontiguousarray(np.logical_not(inpaintloc)).ravel()
        assert(c['inpaint_nloc'].shape == (n_image,))

        c['f'] = np.zeros((n_image, l_labels), order='C')
        clip_hardi_data(data)
        loglog_data = np.log(-np.log(data))
        c['f'][:] = loglog_data.reshape(-1, l_labels)
        f_mean = np.einsum('ik,k->i', c['f'], c['b'])/(4*np.pi)
        c['f'] -= f_mean[:,None]

class ModelHARDI_SHM(ModelHARDI):
    "Base class for HARDI reconstr. using spherical harmonics."
    def __init__(self, *args):
        ModelHARDI.__init__(self, *args)

        c = self.constvars

        sampling_matrix = self.model_params['sampling_matrix']
        c['Y'] = np.zeros(sampling_matrix.shape, order='C')
        c['Y'][:] = sampling_matrix
        l_shm = c['Y'].shape[1]
        c['l_shm'] = l_shm

        c['M'] = self.model_params['model_matrix']
        assert(c['M'].size == c['l_shm'])
        c['YM'] = np.einsum('lk,k->lk', c['Y'], c['M'])

        logging.info("HARDI PDHG setup ({l_labels} labels, {l_shm} shm; " \
                     "img: {imagedims}; lambda={lbd:.3g}) ready.".format(
                         lbd=c['lbd'],
                         l_labels=c['l_labels'],
                         l_shm=c['l_shm'],
                         imagedims="x".join(map(str,c['imagedims']))))
