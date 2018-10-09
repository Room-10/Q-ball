
import logging
import numpy as np

from opymize import Variable
from opymize.functionals import SplitSum, PositivityFct, ZeroFct, \
                                IndicatorFct, AffineFct, L1Norms, ConstrainFct
from opymize.linear import BlockOp, ScaleOp, GradientOp, IndexedMultAdj, \
                           MatrixMultR, MatrixMultRBatched, DiagMatrixMultR

from qball.tools import normalize_odf
from qball.models import ModelHARDI
from qball.operators.pos_ssd import PosSSD

class Model(ModelHARDI):
    name = "n_w_tvw"

    def __init__(self, *args, dataterm="W1", gradnorm="frobenius", **kwargs):
        ModelHARDI.__init__(self, *args, **kwargs)
        self.gradnorm = 'nuclear' if gradnorm == "spectral" else "frobenius"

        c = self.constvars
        imagedims = c['imagedims']
        n_image = c['n_image']
        d_image = c['d_image']
        l_labels = c['l_labels']
        m_gradients = c['m_gradients']
        s_manifold = c['s_manifold']

        xvars = [('u', (n_image, l_labels)),
                 ('w', (m_gradients, n_image, d_image, s_manifold))]
        yvars = [('p', (n_image, d_image, l_labels)),
                 ('g', (m_gradients, n_image, d_image, s_manifold)),
                 ('q', (n_image,))]

        self.dataterm = dataterm
        if self.dataterm == "W1":
            xvars.append(('w0', (m_gradients, n_image, s_manifold)))
            yvars.append(('p0', (n_image, l_labels)))
            yvars.append(('g0', (m_gradients, n_image, s_manifold)))
        elif self.dataterm != "quadratic":
            raise Exception("Dataterm unknown: %s" % self.dataterm)

        self.x = Variable(*xvars)
        self.y = Variable(*yvars)

        # start with a uniform distribution in each voxel
        self.state = (self.x.new(), self.y.new())
        x = self.x.vars(self.state[0], named=True)
        x['u'][:] = 1.0/np.einsum('k->', c['b'])

        f = self.data.odf
        f_flat = f.reshape(-1, l_labels).T
        f = np.array(f_flat.reshape((l_labels,) + imagedims), order='C')
        normalize_odf(f, c['b'])
        self.f = np.array(f.reshape(l_labels, -1).T, order='C')

        logging.info("HARDI setup ({l_labels} labels; " \
                     "img: {imagedims}; lambda={lbd:.3g}) ready.".format(
                         lbd=c['lbd'],
                         l_labels=c['l_labels'],
                         imagedims="x".join(map(str,c['imagedims']))))

    def setup_solver_pdhg(self):
        x, y = self.x.vars(named=True), self.y.vars(named=True)
        c = self.constvars
        imagedims = c['imagedims']
        n_image = c['n_image']
        d_image = c['d_image']
        l_labels = c['l_labels']
        m_gradients = c['m_gradients']
        s_manifold = c['s_manifold']

        GradOp = GradientOp(imagedims, l_labels, weights=c['b'])

        PBLinOp = IndexedMultAdj(l_labels, d_image*n_image, c['P'], c['B'])
        AMult = MatrixMultRBatched(n_image*d_image, c['A'])

        bMult = MatrixMultR(n_image, c['b_precond']*c['b'][:,None])

        l1norms = L1Norms(m_gradients*n_image, (d_image, s_manifold), c['lbd'], self.gradnorm)

        if self.dataterm == "W1":
            self.pdhg_G = SplitSum([
                PositivityFct(x['u']['size']), # \delta_{u >= 0}
                ZeroFct(x['w']['size']),       # 0
                ZeroFct(x['w0']['size']),      # 0
            ])

            PBLinOp0 = IndexedMultAdj(l_labels, n_image, c['P'], c['B'])
            AMult0 = MatrixMultRBatched(n_image, c['A'])
            bMult0 = DiagMatrixMultR(n_image, c['b'])

            self.pdhg_linop = BlockOp([
                [GradOp, PBLinOp,        0], # p = diag(b)Du - P'B'w
                [     0,   AMult,        0], # g = A'w
                [ bMult,       0,        0], # q = <b,u>
                [bMult0,       0, PBLinOp0], # p0 = diag(b) u - P'B'w0
                [     0,       0,   AMult0]  # g0 = A'w0
            ])

            diag_b_f = np.einsum('ik,k->ik', self.f, c['b'])
            dataterm = ConstrainFct(c['inpaint_nloc'], diag_b_f)

            l1norms0 = L1Norms(m_gradients*n_image, (1, s_manifold), 1.0, "frobenius")

            self.pdhg_F = SplitSum([
                IndicatorFct(y['p']['size']),   # \delta_{p = 0}
                l1norms,                        # lbd*\sum_ji |g[j,i,:,:]|_nuc
                IndicatorFct(y['q']['size'], c1=c['b_precond']), # \delta_{q = 1}
                dataterm,                       # \delta_{p0 = diag(b)f}
                l1norms0,                       # \sum_ji |g0[j,i,:]|_2
            ])
        elif self.dataterm == "quadratic":
            dataterm = PosSSD(self.f, vol=c['b'], mask=c['inpaint_nloc'])
            self.pdhg_G = SplitSum([
                dataterm,                   # 0.5*<u-f,u-f>_b + \delta_{u >= 0}
                ZeroFct(x['w']['size']),    # 0
            ])

            self.pdhg_linop = BlockOp([
                [GradOp, PBLinOp], # p = diag(b)Du - P'B'w
                [     0,   AMult], # g = A'w
                [ bMult,       0], # q = <b,u>
            ])

            self.pdhg_F = SplitSum([
                IndicatorFct(y['p']['size']),   # \delta_{p = 0}
                l1norms,                        # lbd*\sum_ji |g[j,i,:,:]|_nuc
                IndicatorFct(y['q']['size'], c1=c['b_precond']), # \delta_{q = 1}
            ])
