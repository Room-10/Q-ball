
import logging
import numpy as np
import cvxpy as cvx

from opymize import Variable
from opymize.functionals import SplitSum, SSD, PositivityFct, \
                                ZeroFct, IndicatorFct, L1Norms
from opymize.linear import BlockOp, ScaleOp, GradientOp, \
                           MatrixMultR, TangledMatrixMultR

from qball.models import ModelHARDI_SHM
from qball.tools.shm import sym_shm_sample_grad
from qball.tools.cvx import cvxVariable, sparse_div_op, cvxOp

class Model(ModelHARDI_SHM):
    name = "sh_l_tvw2"

    def __init__(self, *args, gradnorm="frobenius", **kwargs):
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

        self.x = Variable(
            ('u1', (n_image, l_labels)),
            ('u2', (n_image, l_labels)),
            ('v', (n_image, l_shm)),
            ('w', (m_gradients, n_image, d_image, s_manifold)),
        )

        self.y = Variable(
            ('p', (n_image, d_image, l_shm)),
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

        c['G'] = np.zeros((m_gradients, s_manifold, l_shm), order='C')
        c['G'][:] = sym_shm_sample_grad(c['Y'], self.data.b_sph.v_grad)

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

        dataterm = SSD(c['f'], vol=c['b'], mask=c['inpaint_nloc'])

        self.pdhg_G = SplitSum([
            PositivityFct(x['u1']['size']), # \delta_{u1 >= 0}
            dataterm,                       # 0.5*<u2-f,u2-f>
            ZeroFct(x['v']['size']),        # 0
            ZeroFct(x['w']['size'])         # 0
        ])

        GradOp = GradientOp(imagedims, l_shm)

        GMult = TangledMatrixMultR(n_image*d_image, c['G'][:,:,None,:])

        bMult = MatrixMultR(n_image, c['b_precond']*c['b'][:,None])
        YMult = MatrixMultR(n_image, c['Y'], trans=True)
        YMMult = MatrixMultR(n_image, c['YM'], trans=True)

        m_u = ScaleOp(x['u1']['size'], -1)
        m_w = ScaleOp(x['w']['size'], -1)

        self.pdhg_linop = BlockOp([
            [     0,    0, GradOp, GMult], # p = Dv + G'w
            [     0,    0,      0,   m_w], # g = -w
            [ bMult,    0,      0,     0], # q0 = <b,u1>
            [   m_u,    0,  YMult,     0], # q1 = Yv - u1
            [     0,  m_u, YMMult,     0]  # q2 = YMv - u2
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
            ('p', (l_shm, d_image, n_image)),
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

        self.cvx_obj = cvx.Maximize(
              0.5*cvx.sum(cvx.diag(c['b'])*cvx.square(c['f'].T))
            - 0.5*cvx.sum(
                cvx.diag(1.0/c['b'])*cvx.square(q2 + cvx.diag(c['b'])*c['f'].T)
            ) - cvx.sum(q0))

        div_op = sparse_div_op(imagedims)

        self.cvx_dual = True
        self.cvx_constr = []

        # u1_constr
        for i in range(n_image):
            self.cvx_constr.append(c['b']*q0[i] - q1[:,i] >= 0)

        # v_constr
        for i in range(n_image):
            for k in range(l_shm):
                Yk = cvx.vec(c['Y'][:,k])
                self.cvx_constr.append(
                    Yk.T*(c['M'][k]*q2[:,i] + q1[:,i]) \
                        - cvxOp(div_op, p[k], i) == 0)

        # w_constr
        for j in range(m_gradients):
            Gj = c['G'][j,:,:]
            for i in range(n_image):
                for t in range(d_image):
                    for l in range(s_manifold):
                        self.cvx_constr.append(
                            g[i][j][l,t] == sum([Gj[l,k]*p[k][t,i] \
                                                 for k in range(l_shm)]))

        # additional inequality constraints
        for i in range(n_image):
            for j in range(m_gradients):
                self.cvx_constr.append(cvx.norm(g[i][j], 2) <= c['lbd'])

