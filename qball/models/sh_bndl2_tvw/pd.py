
from qball.models.base import ModelHARDI_SHM
from qball.tools.bounds import compute_hardi_bounds
from qball.operators.bndl2 import BndSSD

import numpy as np

import logging

from opymize import BlockVar
from opymize.functionals import SplitSum, SSD, PositivityFct, \
                                ZeroFct, IndicatorFct, L1Norms
from opymize.linear import BlockOp, ScaleOp, GradientOp, IndexedMultAdj, \
                           MatrixMultR, MatrixMultRBatched

def fit_hardi_qball(data, model_params, solver_params):
    solver = MyModel(data, model_params)
    details = solver.solve(**solver_params)
    return solver.state, details

class MyModel(ModelHARDI_SHM):
    def __init__(self, *args):
        ModelHARDI_SHM.__init__(self, *args)

        c = self.constvars
        e = self.extravars

        imagedims = c['imagedims']
        n_image = c['n_image']
        d_image = c['d_image']
        l_labels = c['l_labels']
        m_gradients = c['m_gradients']
        s_manifold = c['s_manifold']
        l_shm = c['l_shm']

        alpha = self.model_params.get('conf_lvl', 0.9)
        compute_hardi_bounds(self.data, alpha=alpha)
        _, f1, f2 = self.data['bounds']
        f1, f2 = [np.array(a.T, order='C') for a in [f1,f2]]

        self.x = BlockVar(
            ('u1', (n_image, l_labels)),
            ('u2', (n_image, l_labels)),
            ('v', (n_image, l_shm)),
            ('w', (m_gradients, n_image, d_image, s_manifold)),
        )

        self.y = BlockVar(
            ('p', (n_image, d_image, l_labels)),
            ('g', (m_gradients, n_image, d_image, s_manifold)),
            ('q0', (n_image,)),
            ('q1', (n_image, l_labels)),
            ('q2', (n_image, l_labels)),
        )

        # start with a uniform distribution in each voxel
        u1k, u2k, vk, wk = self.x.vars()
        u1k[:] = 1.0/np.einsum('k->', e['b_sph'].b)
        vk[:,0] = .5 / np.sqrt(np.pi)

        N_u = self.x['u1'].size
        N_v = self.x['v'].size
        N_w = self.x['w'].size
        N_p = self.y['p'].size
        N_q0 = self.y['q0'].size

        dataterm = BndSSD(f1, f2, vol=e['b_sph'].b, mask=c['inpaint_nloc'])

        self.G = SplitSum([
            PositivityFct(N_u), # \delta_{u1 >= 0}
            dataterm,       # 0.5*|max(0, f1 - u2)|^2 + 0.5*|max(0, u2 - f2)|^2
            ZeroFct(N_v),       # 0
            ZeroFct(N_w)        # 0
        ])

        GradOp = GradientOp(imagedims, l_labels, weights=e['b_sph'].b)

        PBLinOp = IndexedMultAdj(l_labels, d_image*n_image, e['b_sph'].P, e['b_sph'].B)
        AMult = MatrixMultRBatched(n_image*d_image, e['b_sph'].A)

        bMult = MatrixMultR(n_image, e['b_sph'].b_precond*e['b_sph'].b[:,None])
        YMult = MatrixMultR(n_image, c['Y'], trans=True)
        YMMult = MatrixMultR(n_image, c['YM'], trans=True)

        m_u = ScaleOp(N_u, -1)

        self.linop = BlockOp([
            [GradOp,   0,      0, PBLinOp], # p = diag(b)Du1 - P'B'w
            [     0,   0,      0,   AMult], # g = A'w
            [ bMult,   0,      0,       0], # q0 = <b,u1>
            [   m_u,   0,  YMult,       0], # q1 = Yv - u1
            [     0, m_u, YMMult,       0]  # q2 = YMv - u2
        ])

        gradnorm = self.model_params.get('gradnorm', "frobenius")
        matrixnorm = 'nuclear' if gradnorm == "spectral" else "frobenius"
        l1norms = L1Norms(m_gradients*n_image, (d_image, s_manifold), c['lbd'], matrixnorm)

        self.F = SplitSum([
            IndicatorFct(N_p),                      # \delta_{p = 0}
            l1norms,                                # lbd*\sum_ji |g[j,i,:,:]|_nuc
            IndicatorFct(N_q0, c1=e['b_sph'].b_precond), # \delta_{q0 = 1}
            IndicatorFct(N_u),                      # \delta_{q1 = 0}
            IndicatorFct(N_u)                       # \delta_{q2 = 0}
        ])
