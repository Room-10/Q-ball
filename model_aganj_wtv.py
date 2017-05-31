
import numpy as np
from dipy.reconst.shm import CsaOdfModel

#from solve_cvx import w1_tv_regularization
from solve_cuda import w1_tv_regularization

class AganjWassersteinModel(CsaOdfModel):
    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        sh_coef = CsaOdfModel._get_shm_coef(self, data, mask)
        f = np.dot(sh_coef, self.B.T).T
        u, v = w1_tv_regularization(f, self.gtab, sampling_matrix=self.B)
        sh_coef = v
        return sh_coef
