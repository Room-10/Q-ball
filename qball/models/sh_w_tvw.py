
import logging
import numpy as np
import cvxpy as cvx

from opymize import Variable
from opymize.functionals import SplitSum, PositivityFct, ZeroFct, \
                                IndicatorFct, AffineFct, L1Norms, ConstrainFct
from opymize.linear import BlockOp, ScaleOp, GradientOp, IndexedMultAdj, \
                           MatrixMultR, MatrixMultRBatched, DiagMatrixMultR

from qball.tools import normalize_odf
from qball.tools.cvx import cvxVariable, sparse_div_op, cvxOp
from qball.models import ModelHARDI_SHM
from qball.operators.pos_ssd import PosSSD

class Model(ModelHARDI_SHM):
    name = "sh_w_tvw"

    def __init__(self, *args, dataterm="W1", gradnorm="frobenius", **kwargs):
        ModelHARDI_SHM.__init__(self, *args, **kwargs)
        self.gradnorm = 'nuclear' if gradnorm == "spectral" else "frobenius"

        c = self.constvars
        b_sph  = self.data.b_sph
        imagedims = c['imagedims']
        n_image = c['n_image']
        d_image = c['d_image']
        l_labels = c['l_labels']
        m_gradients = c['m_gradients']
        s_manifold = c['s_manifold']
        l_shm = c['l_shm']

        xvars = [('u', (n_image, l_labels)),
                 ('v', (n_image, l_shm)),
                 ('w', (m_gradients, n_image, d_image, s_manifold))]
        yvars = [('p', (n_image, d_image, l_labels)),
                 ('g', (m_gradients, n_image, d_image, s_manifold)),
                 ('q0', (n_image,)),
                 ('q1', (n_image, l_labels))]

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
        x['v'][:,0] = .5 / np.sqrt(np.pi)

        f = self.data.odf
        f_flat = f.reshape(-1, l_labels).T
        f = np.array(f_flat.reshape((l_labels,) + imagedims), order='C')
        normalize_odf(f, c['b'])
        self.f = np.array(f.reshape(l_labels, -1).T, order='C')

    def setup_solver_pdhg(self):
        x, y = self.x.vars(named=True), self.y.vars(named=True)
        c = self.constvars
        imagedims = c['imagedims']
        n_image = c['n_image']
        d_image = c['d_image']
        l_labels = c['l_labels']
        m_gradients = c['m_gradients']
        s_manifold = c['s_manifold']
        l_shm = c['l_shm']

        GradOp = GradientOp(imagedims, l_labels, weights=c['b'])

        PBLinOp = IndexedMultAdj(l_labels, d_image*n_image, c['P'], c['B'])
        AMult = MatrixMultRBatched(n_image*d_image, c['A'])

        bMult = MatrixMultR(n_image, c['b_precond']*c['b'][:,None])

        YMult = MatrixMultR(n_image, c['Y'], trans=True)
        m_u = ScaleOp(x['u']['size'], -1)

        l1norms = L1Norms(m_gradients*n_image, (d_image, s_manifold), c['lbd'], self.gradnorm)

        if self.dataterm == "W1":
            self.pdhg_G = SplitSum([
                PositivityFct(x['u']['size']), # \delta_{u >= 0}
                ZeroFct(x['v']['size']),       # 0
                ZeroFct(x['w']['size']),       # 0
                ZeroFct(x['w0']['size']),      # 0
            ])

            PBLinOp0 = IndexedMultAdj(l_labels, n_image, c['P'], c['B'])
            AMult0 = MatrixMultRBatched(n_image, c['A'])
            bMult0 = DiagMatrixMultR(n_image, c['b'])

            self.pdhg_linop = BlockOp([
                [GradOp,      0, PBLinOp,        0], # p = diag(b)Du - P'B'w
                [     0,      0,   AMult,        0], # g = A'w
                [ bMult,      0,       0,        0], # q0 = <b,u>
                [   m_u,  YMult,       0,        0], # q1 = Yv - u
                [bMult0,      0,       0, PBLinOp0], # p0 = diag(b) u - P'B'w0
                [     0,      0,       0,   AMult0]  # g0 = A'w0
            ])

            diag_b_f = np.einsum('ik,k->ik', self.f, c['b'])
            dataterm = ConstrainFct(c['inpaint_nloc'], diag_b_f)

            l1norms0 = L1Norms(m_gradients*n_image, (1, s_manifold), 1.0, "frobenius")

            self.pdhg_F = SplitSum([
                IndicatorFct(y['p']['size']),   # \delta_{p = 0}
                l1norms,                        # lbd*\sum_ji |g[j,i,:,:]|_nuc
                IndicatorFct(y['q0']['size'], c1=c['b_precond']), # \delta_{q0 = 1}
                IndicatorFct(x['u']['size']),   # \delta_{q1 = 0}
                dataterm,                       # \delta_{p0 = diag(b)f}
                l1norms0,                       # \sum_ji |g0[j,i,:]|_2
            ])
        elif self.dataterm == "quadratic":
            dataterm = PosSSD(self.f, vol=c['b'], mask=c['inpaint_nloc'])
            self.pdhg_G = SplitSum([
                dataterm,                   # 0.5*<u-f,u-f>_b + \delta_{u >= 0}
                ZeroFct(x['v']['size']),    # 0
                ZeroFct(x['w']['size']),    # 0
            ])

            self.pdhg_linop = BlockOp([
                [GradOp,      0, PBLinOp],  # p = diag(b)Du - P'B'w
                [     0,      0,   AMult],  # g = A'w
                [ bMult,      0,       0],  # q0 = <b,u>
                [   m_u,  YMult,       0],  # q1 = Yv - u
            ])

            self.pdhg_F = SplitSum([
                IndicatorFct(y['p']['size']),   # \delta_{p = 0}
                l1norms,                        # lbd*\sum_ji |g[j,i,:,:]|_nuc
                IndicatorFct(y['q0']['size'], c1=c['b_precond']), # \delta_{q0 = 1}
                IndicatorFct(x['u']['size']),   # \delta_{q1 = 0}
            ])

    def setup_solver_cvx(self):
        if self.dataterm != "W1":
            raise Exception("Only W1 dataterm is implemented in CVX")

        c = self.constvars
        imagedims = c['imagedims']
        n_image = c['n_image']
        d_image = c['d_image']
        l_labels = c['l_labels']
        m_gradients = c['m_gradients']
        s_manifold = c['s_manifold']
        l_shm = c['l_shm']

        f_flat = self.data.odf.reshape(-1, l_labels).T
        f = np.array(f_flat.reshape((l_labels,) + imagedims), order='C')
        normalize_odf(f, c['b'])
        f_flat = f.reshape(l_labels, n_image)

        self.cvx_x = Variable(
            ('p', (l_labels, d_image, n_image)),
            ('g', (n_image, m_gradients, s_manifold, d_image)),
            ('q0', (n_image,)),
            ('q1', (l_labels, n_image)),
            ('p0', (l_labels, n_image)),
            ('g0', (n_image, m_gradients, s_manifold)),
        )

        self.cvx_y = Variable(
            ('u', (n_image, l_labels)),
            ('v', (n_image, l_shm)),
            ('w', (m_gradients, n_image, d_image, s_manifold)),
            ('w0', (m_gradients, n_image, s_manifold)),
            ('misc', (n_image*m_gradients,)),
        )

        p, g, q0, q1, p0, g0 = [cvxVariable(*a['shape']) for a in self.cvx_x.vars()]
        self.cvx_vars = p + sum(g,[]) + [q0,q1,p0] + g0

        self.cvx_obj = cvx.Maximize(
            - cvx.vec(f_flat).T*cvx.vec(cvx.diag(c['b'])*p0)
            - cvx.sum(q0))

        div_op = sparse_div_op(imagedims)

        self.cvx_dual = True
        self.cvx_constr = []

        # u_constr
        for i in range(n_image):
            for k in range(l_labels):
                self.cvx_constr.append(
                    c['b'][k]*(q0[i] + p0[k,i] \
                        - cvxOp(div_op, p[k], i)) - q1[k,i] >= 0)

        # v_constr
        for i in range(n_image):
            for k in range(l_shm):
                Yk = cvx.vec(c['Y'][:,k])
                self.cvx_constr.append(-Yk.T*q1[:,i] == 0)

        # w_constr
        for j in range(m_gradients):
            Aj = c['A'][j,:,:]
            Bj = c['B'][j,:,:]
            Pj = c['P'][j,:]
            for i in range(n_image):
                for t in range(d_image):
                    for l in range(s_manifold):
                        self.cvx_constr.append(
                            Aj*g[i][j][l,t] == sum([Bj[l,m]*p[Pj[m]][t,i] \
                                                    for m in range(3)]))

        # w0_constr
        for j in range(m_gradients):
            Aj = c['A'][j,:,:]
            Bj = c['B'][j,:,:]
            Pj = c['P'][j,:]
            for i in range(n_image):
                self.cvx_constr.append(Aj*g0[i][j,:].T == Bj*p0[Pj,i])

        # additional inequality constraints
        for i in range(n_image):
            for j in range(m_gradients):
                self.cvx_constr.append(cvx.norm(g[i][j], 2) <= c['lbd'])
                self.cvx_constr.append(cvx.norm(g0[i][j,:], 2) <= 1.0)
