
import numpy as np
from dipy.reconst.shm import CsaOdfModel

class AganjWassersteinModel(CsaOdfModel):
    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        sh_coef = CsaOdfModel._get_shm_coef(self, data, mask)
        f = np.zeros((self.B.shape[0],) + sh_coef.shape[:-1], order='C')
        f[:] = np.dot(sh_coef, self.B.T).T
        from solve_qbshm_cuda import w1_tv_regularization
        pd_state, details = w1_tv_regularization(f, self.gtab,
            sampling_matrix=self.B, use_gpu=False)
        sh_coef = pd_state[1].T.reshape(sh_coef.shape)
        return sh_coef

class AganjWassersteinModelGPU(CsaOdfModel):
    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        sh_coef = CsaOdfModel._get_shm_coef(self, data, mask)
        f = np.zeros((self.B.shape[0],) + sh_coef.shape[:-1], order='C')
        f[:] = np.dot(sh_coef, self.B.T).T
        from solve_qbshm_cuda import w1_tv_regularization
        pd_state, details = w1_tv_regularization(f, self.gtab,
            sampling_matrix=self.B, use_gpu=True)
        sh_coef = pd_state[1].T.reshape(sh_coef.shape)
        return sh_coef

class AganjWassersteinModelCVX(CsaOdfModel):
    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        sh_coef = CsaOdfModel._get_shm_coef(self, data, mask)
        f = np.zeros((self.B.shape[0],) + sh_coef.shape[:-1], order='C')
        f[:] = np.dot(sh_coef, self.B.T).T
        from solve_qbshm_cvx import w1_tv_regularization
        pd_state, details = w1_tv_regularization(f, self.gtab, sampling_matrix=self.B)
        sh_coef = pd_state[1].T.reshape(sh_coef.shape)
        return sh_coef