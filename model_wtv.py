
import numpy as np
from dipy.reconst.shm import QballBaseModel

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
        data = data[..., self._where_dwi]
        data = data.clip(self.min, self.max)
        Minv = np.zeros(self._fit_matrix.shape)
        Minv[1:] = 1.0/self._fit_matrix[1:]
        from solve_hardi_cuda import l2_w1tv_fitting
        pd_state, details = l2_w1tv_fitting(data, self.gtab, self.B, Minv,
            use_gpu=False)
        sh_coef = pd_state[2].T.reshape(data.shape[:-1]+(self.B.shape[1],))
        sh_coef[..., 0] = self._n0_const
        return sh_coef

class WassersteinModelGPU(QballBaseModel):
    """ Implementation of Wasserstein-TV model """
    min = .001
    max = .999
    _n0_const = .5 / np.sqrt(np.pi)

    def _set_fit_matrix(self, B, L, F, smooth):
        """ The fit matrix describes the forward model. """
        self._fit_matrix = (F * L) / (8 * np.pi)

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        data = data[..., self._where_dwi]
        data = data.clip(self.min, self.max)
        Minv = np.zeros(self._fit_matrix.shape, order='C')
        Minv[1:] = 1.0/self._fit_matrix[1:]
        from solve_hardi_cuda import l2_w1tv_fitting
        pd_state, details = l2_w1tv_fitting(data, self.gtab, self.B, Minv,
            use_gpu=True)
        sh_coef = pd_state[2].T.reshape(data.shape[:-1]+(self.B.shape[1],))
        sh_coef[..., 0] = self._n0_const
        return sh_coef

class WassersteinModelCVX(QballBaseModel):
    """ Implementation of Wasserstein-TV model """
    min = .001
    max = .999
    _n0_const = .5 / np.sqrt(np.pi)

    def _set_fit_matrix(self, B, L, F, smooth):
        """ The fit matrix describes the forward model. """
        self._fit_matrix = (F * L) / (8 * np.pi)

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        data = data[..., self._where_dwi]
        data = data.clip(self.min, self.max)
        Minv = np.zeros(self._fit_matrix.shape)
        Minv[1:] = 1.0/self._fit_matrix[1:]
        from solve_cvx import l2_w1tv_fitting
        u, v = l2_w1tv_fitting(data, self.gtab, self.B, Minv)
        sh_coef = v
        return sh_coef
