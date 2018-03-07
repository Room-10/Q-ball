
from qball.tools import normalize_odf
from qball.tools.blocks import BlockVar
from qball.models.base import ModelHARDI
from qball.operators.pbmult import PBMult
from qball.operators.pos_ssd import PosSSD

import numpy as np

import logging

from opymize.functionals import SplitSum, PositivityFct, ZeroFct, \
                                IndicatorFct, AffineFct, L1Norms, ConstrainFct
from opymize.linear import BlockOp, ScaleOp, GradientOp, \
                           MatrixMultR, MatrixMultRBatched, DiagMatrixMultR

def qball_regularization(data, model_params, solver_params):
    solver = MyPDHGModel(data, model_params)
    details = solver.solve(**solver_params)
    return solver.state, details

class MyPDHGModel(ModelHARDI):
    def __init__(self, *args):
        ModelHARDI.__init__(self, *args)

        c = self.constvars
        e = self.extravars

        imagedims = c['imagedims']
        n_image = c['n_image']
        d_image = c['d_image']
        l_labels = c['l_labels']
        m_gradients = c['m_gradients']
        s_manifold = c['s_manifold']

        f = self.data['odf']
        f_flat = f.reshape(-1, l_labels).T
        f = np.array(f_flat.reshape((l_labels,) + imagedims), order='C')
        normalize_odf(f, e['b_sph'].b)
        f = np.array(f.reshape(l_labels, -1).T, order='C')

        xvars = [('u', (n_image, l_labels)),
                 ('w', (m_gradients, n_image, d_image, s_manifold))]
        yvars = [('p', (n_image, d_image, l_labels)),
                 ('g', (m_gradients, n_image, d_image, s_manifold)),
                 ('q', (n_image,))]

        dataterm = self.model_params.get('dataterm', "W1")
        if dataterm == "W1":
            xvars.append(('w0', (m_gradients, n_image, s_manifold)))
            yvars.append(('p0', (n_image, l_labels)))
            yvars.append(('g0', (m_gradients, n_image, s_manifold)))
        elif dataterm != "quadratic":
            raise Exception("Dataterm unknown: %s" % dataterm)

        self.x = BlockVar(*xvars)
        self.y = BlockVar(*yvars)

        # start with a uniform distribution in each voxel
        self.x['u'][:] = 1.0/np.einsum('k->', e['b_sph'].b)

        N_u = self.x['u'].size
        N_w = self.x['w'].size
        N_p = self.y['p'].size
        N_q0 = self.y['q'].size

        GradOp = GradientOp(imagedims, l_labels, weights=e['b_sph'].b)

        PBLinOp = PBMult(l_labels, d_image*n_image, e['b_sph'].P, e['b_sph'].B)
        AMult = MatrixMultRBatched(n_image*d_image, e['b_sph'].A)

        bMult = MatrixMultR(n_image, e['b_sph'].b_precond*e['b_sph'].b[:,None])

        gradnorm = self.model_params.get('gradnorm', "frobenius")
        matrixnorm = 'nuclear' if gradnorm == "spectral" else "frobenius"
        l1norms = L1Norms(m_gradients*n_image, (d_image, s_manifold), c['lbd'], matrixnorm)

        if dataterm == "W1":
            N_w0 = self.x['w0'].size
            N_p0 = self.y['p0'].size
            self.G = SplitSum([
                PositivityFct(N_u), # \delta_{u >= 0}
                ZeroFct(N_w),       # 0
                ZeroFct(N_w0),      # 0
            ])

            PBLinOp0 = PBMult(l_labels, n_image, e['b_sph'].P, e['b_sph'].B)
            AMult0 = MatrixMultRBatched(n_image, e['b_sph'].A)
            bMult0 = DiagMatrixMultR(n_image, e['b_sph'].b)

            self.linop = BlockOp([
                [GradOp, PBLinOp,        0], # p = diag(b)Du - P'B'w
                [     0,   AMult,        0], # g = A'w
                [ bMult,       0,        0], # q = <b,u>
                [bMult0,       0, PBLinOp0], # p0 = diag(b) u - P'B'w0
                [     0,       0,   AMult0]  # g0 = A'w0
            ])

            diag_b_f = np.einsum('ik,k->ik', f, e['b_sph'].b)
            dataterm = ConstrainFct(c['inpaint_nloc'], diag_b_f)

            l1norms0 = L1Norms(m_gradients*n_image, (1, s_manifold), 1.0, "frobenius")

            self.F = SplitSum([
                IndicatorFct(N_p),                      # \delta_{p = 0}
                l1norms,                                # lbd*\sum_ji |g[j,i,:,:]|_nuc
                IndicatorFct(N_q0, c1=e['b_sph'].b_precond), # \delta_{q0 = 1}
                dataterm,                               # \delta_{p0 = diag(b)f}
                l1norms0,                               # \sum_ji |g0[j,i,:]|_2
            ])
        elif dataterm == "quadratic":
            dataterm = PosSSD(f, vol=e['b_sph'].b, mask=c['inpaint_nloc'])
            self.G = SplitSum([
                dataterm,       # 0.5*<u-f,u-f>_b + \delta_{u >= 0}
                ZeroFct(N_w),   # 0
            ])

            self.linop = BlockOp([
                [GradOp, PBLinOp], # p = diag(b)Du - P'B'w
                [     0,   AMult], # g = A'w
                [ bMult,       0], # q = <b,u>
            ])

            self.F = SplitSum([
                IndicatorFct(N_p),                      # \delta_{p = 0}
                l1norms,                                # lbd*\sum_ji |g[j,i,:,:]|_nuc
                IndicatorFct(N_q0, c1=e['b_sph'].b_precond), # \delta_{q0 = 1}
            ])
