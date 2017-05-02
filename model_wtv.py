
import itertools
import numpy as np
import cvxpy as cvx
from dipy.reconst.shm import QballBaseModel, smooth_pinv

from manifold_sphere import load_sphere
from tools_cvx import cvxVariable, sparse_div_op, cvxOp

class WassersteinModel(QballBaseModel):
    """ Implementation of Wasserstein-TV model """
    min = .001
    max = .999
    _n0_const = .5 / np.sqrt(np.pi)

    def _set_fit_matrix(self, B, L, F, smooth):
        """ The fit matrix describes the forward model. """
        self._fit_matrix = (F * L) / (8 * np.pi)

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        b_vecs = self.gtab.bvecs[self.gtab.bvals > 0,...].T
        b_sph = load_sphere(vecs=b_vecs)

        l_labels, l_shm = self.B.shape
        m_gradients = b_sph.mdims['m_gradients']
        imagedims = data.shape[:-1]
        d_image = len(imagedims)
        n_image = np.prod(imagedims)
        lbd = 1e+5

        data = data[..., self._where_dwi]
        data = data.clip(self.min, self.max)
        F_data = np.log(-np.log(data)).reshape(-1, l_labels).T
        F_mean = np.einsum('ki,k->i', F_data, b_sph.b)/(4*np.pi)
        F_data -= F_mean

        p  = cvxVariable(l_labels, d_image, n_image)
        g  = cvxVariable(n_image, m_gradients, d_image, 2)
        q0 = cvxVariable(n_image)
        q1 = cvxVariable(l_labels, n_image)
        q2 = cvxVariable(l_labels, n_image)

        Y = self.B
        Minv = np.zeros(self._fit_matrix.shape)
        Minv[1:] = 1.0/self._fit_matrix[1:]

        obj = cvx.Maximize(
            - 0.25*cvx.sum_entries(cvx.square(q1))
            - cvx.vec(F_data).T*cvx.vec(q1)
            - cvx.sum_entries(q0)
        )

        div_op = sparse_div_op(imagedims)

        constraints = []
        for i in range(n_image):
            for j in range(m_gradients):
                constraints.append(cvx.norm(g[i][j], 2) <= lbd)

        w_constr = []
        for j in range(m_gradients):
            Aj = b_sph.A[j,:,:]
            Bj = b_sph.B[j,:,:]
            Pj = b_sph.P[j,:]
            for i in range(n_image):
                for t in range(d_image):
                    w_constr.append(
                        Aj*g[i][j][t,:].T == sum([Bj[:,m]*p[Pj[m]][t,i] for m in range(3)])
                    )
        constraints += w_constr

        u_constr = []
        for k in range(l_labels):
            for i in range(n_image):
                u_constr.append(
                   b_sph.b[k]*(q0[i] - cvxOp(div_op, p[k], i)) - q2[k,i] >= 0
                )
        constraints += u_constr

        v_constr = []
        for k in range(l_shm):
            for i in range(n_image):
                Yk = cvx.vec(Y[:,k])
                v_constr.append(
                    Yk.T*(Minv[k]*q1[:,i] + q2[:,i]) == 0
                )
        constraints += v_constr

        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True)

        v = np.zeros((l_shm, n_image), order='C')
        for k in range(l_shm):
            for i in range(n_image):
                v[k,i] = -v_constr[k*n_image+i].dual_value

        sh_coef = v.T.reshape(imagedims + (l_shm,))
        sh_coef[..., 0] = self._n0_const
        return sh_coef
