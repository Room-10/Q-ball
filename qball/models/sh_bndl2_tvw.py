
import logging
import numpy as np
import cvxpy as cvx

from opymize import Variable
from opymize.functionals import SplitSum, SSD, PositivityFct, \
                                ZeroFct, IndicatorFct, L1Norms
from opymize.linear import BlockOp, ScaleOp, GradientOp, IndexedMultAdj, \
                           MatrixMultR, MatrixMultRBatched

from qball.models import ModelHARDI_SHM
from qball.tools.cvx import cvxVariable, sparse_div_op, cvxOp
from qball.operators.bndl2 import BndSSD

# TODO: inpaint mask

class Model(ModelHARDI_SHM):
    name = "sh_bndl2_tvw"

    def __init__(self, *args, gradnorm="frobenius", conf_lvl=0.9, **kwargs):
        ModelHARDI_SHM.__init__(self, *args, **kwargs)
        self.gradnorm = 'nuclear' if gradnorm == "spectral" else "frobenius"

        c = self.constvars
        imagedims = c['imagedims']
        n_image = c['n_image']
        d_image = c['d_image']
        l_labels = c['l_labels']
        m_gradients = c['m_gradients']
        s_manifold = c['s_manifold']
        l_shm = c['l_shm']

        self.data.init_bounds(conf_lvl)
        _, f1, f2 = self.data.bounds
        c['f1'], c['f2'] = [np.array(a.T, order='C') for a in [f1,f2]]

        self.x = Variable(
            ('u1', (n_image, l_labels)),
            ('u2', (n_image, l_labels)),
            ('v', (n_image, l_shm)),
            ('w', (m_gradients, n_image, d_image, s_manifold)),
        )

        self.y = Variable(
            ('p', (n_image, d_image, l_labels)),
            ('g', (m_gradients, n_image, d_image, s_manifold)),
            ('q0', (n_image,)),
            ('q1', (n_image, l_labels)),
            ('q2', (n_image, l_labels)),
        )

        # start with a uniform distribution in each voxel
        self.state = (self.x.new(), self.y.new())
        u1k, u2k, vk, wk = self.x.vars(self.state[0])
        u1k[:] = 1.0/np.einsum('k->', c['b'])
        vk[:,0] = .5 / np.sqrt(np.pi)

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

        dataterm = BndSSD(c['f1'], c['f2'], vol=c['b'], mask=c['inpaint_nloc'])

        self.pdhg_G = SplitSum([
            PositivityFct(x['u1']['size']), # \delta_{u1 >= 0}
            dataterm,                       # 0.5*|max(0, f1 - u2)|^2 + 0.5*|max(0, u2 - f2)|^2
            ZeroFct(x['v']['size']),        # 0
            ZeroFct(x['w']['size'])         # 0
        ])

        GradOp = GradientOp(imagedims, l_labels, weights=c['b'])

        PBLinOp = IndexedMultAdj(l_labels, d_image*n_image, c['P'], c['B'])
        AMult = MatrixMultRBatched(n_image*d_image, c['A'])

        bMult = MatrixMultR(n_image, c['b_precond']*c['b'][:,None])
        YMult = MatrixMultR(n_image, c['Y'], trans=True)
        YMMult = MatrixMultR(n_image, c['YM'], trans=True)

        m_u = ScaleOp(x['u1']['size'], -1)

        self.pdhg_linop = BlockOp([
            [GradOp,   0,      0, PBLinOp], # p = diag(b)Du1 - P'B'w
            [     0,   0,      0,   AMult], # g = A'w
            [ bMult,   0,      0,       0], # q0 = <b,u1>
            [   m_u,   0,  YMult,       0], # q1 = Yv - u1
            [     0, m_u, YMMult,       0]  # q2 = YMv - u2
        ])

        l1norms = L1Norms(m_gradients*n_image, (d_image, s_manifold), c['lbd'], self.gradnorm)

        self.pdhg_F = SplitSum([
            IndicatorFct(y['p']['size']),   # \delta_{p = 0}
            l1norms,                        # lbd*\sum_ji |g[j,i,:,:]|_nuc
            IndicatorFct(y['q0']['size'], c1=c['b_precond']), # \delta_{q0 = 1}
            IndicatorFct(x['u1']['size']),  # \delta_{q1 = 0}
            IndicatorFct(x['u1']['size'])   # \delta_{q2 = 0}
        ])

    def setup_solver_cvx(self):
        c = self.constvars
        imagedims = c['imagedims']
        n_image = c['n_image']
        d_image = c['d_image']
        l_labels = c['l_labels']
        m_gradients = c['m_gradients']
        s_manifold = c['s_manifold']
        l_shm = c['l_shm']

        self.cvx_x = Variable(
            ('p', (l_labels, d_image, n_image)),
            ('g', (n_image, m_gradients, s_manifold, d_image)),
            ('q0', (n_image,)),
            ('q1', (l_labels, n_image)),
            ('q2', (l_labels, n_image)),
        )

        self.cvx_y = Variable(
            ('u1', (n_image, l_labels)),
            ('v', (n_image, l_shm)),
            ('w', (m_gradients, n_image, d_image, s_manifold)),
            ('misc', (n_image*m_gradients,)),
        )

        p, g, q0, q1, q2 = [cvxVariable(*a['shape']) for a in self.cvx_x.vars()]
        self.cvx_vars = p + sum(g,[]) + [q0,q1,q2]

        fid_fun_dual = 0
        for i in range(n_image):
            for k in range(l_labels):
                fid_fun_dual += -cvx.power(q2[k,i],2)/2 \
                             - cvx.maximum(q2[k,i]*c['f1'][i,k],
                                 q2[k,i]*c['f2'][i,k])

        self.cvx_obj = cvx.Maximize(fid_fun_dual - cvx.sum(q0))

        div_op = sparse_div_op(imagedims)

        self.cvx_dual = True
        self.cvx_constr = []

        # u1_constr
        for i in range(n_image):
            for k in range(l_labels):
                self.cvx_constr.append(
                    c['b'][k]*(q0[i] - cvxOp(div_op, p[k], i)) - q1[k,i] >= 0)

        # v_constr
        for i in range(n_image):
            for k in range(l_shm):
                Yk = cvx.vec(c['Y'][:,k])
                self.cvx_constr.append(Yk.T*(c['M'][k]*q2[:,i] + q1[:,i]) == 0)

        # w_constr
        for j in range(m_gradients):
            Aj = c['A'][j,:,:]
            Bj = c['B'][j,:,:]
            Pj = c['P'][j,:]
            for i in range(n_image):
                for t in range(d_image):
                    for l in range(s_manifold):
                        self.cvx_constr.append(
                            Aj*g[i][j][l,t] == sum([Bj[l,m]*p[Pj[m]][t,i]
                                                    for m in range(3)]))

        # additional inequality constraints
        for i in range(n_image):
            for j in range(m_gradients):
                self.cvx_constr.append(cvx.norm(g[i][j], 2) <= c['lbd'])
