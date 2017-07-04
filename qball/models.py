
import numpy as np

import dipy.core.sphere
from dipy.reconst.odf import OdfFit
from dipy.reconst.shm import QballBaseModel, CsaOdfModel

class n_w_tvw_Model(CsaOdfModel):
    """ Implementation of Wasserstein-TV model from SSVM """
    def fit(self, *args, solver_params={}, sphere=None, **kwargs):
        from qball.solvers.n_w_tvw.cuda import qball_regularization
        if sphere is None:
            b_vecs = self.gtab.bvecs[self.gtab.bvals > 0,...]
            sphere = dipy.core.sphere.Sphere(xyz=b_vecs)
        f = CsaOdfModel.fit(self, *args, **kwargs).odf(sphere)
        f = np.clip(f, 0, np.max(f, -1)[..., None])
        pd_state, details = qball_regularization(f, self.gtab, **solver_params)
        self.solver_state = pd_state
        self.solver_details = details
        l_labels = pd_state[0].shape[0]
        imagedims = pd_state[0].shape[1:]
        u = pd_state[0].reshape(l_labels, -1)
        u = u.T.reshape(imagedims + (l_labels,))
        return TrivialOdfFit(u, sphere)

class TrivialOdfFit(OdfFit):
    def __init__(self, data, sphere):
        self.sphere = sphere
        self.data = data

    def odf(self, sphere=None):
        if sphere is not None and sphere != self.sphere:
            raise Exception("Only original reconstruction sphere supported.")
        else:
            return self.data

class sh_w_tvw_Model(CsaOdfModel):
    """ Implementation of Wasserstein-TV model with SHM regularization """
    def fit(self, *args, solver_engine="cvx", solver_params={}, **kwargs):
        if solver_engine == "cvx":
            from qball.solvers.sh_w_tvw.cvx import qball_regularization
        else:
            from qball.solvers.sh_w_tvw.cuda import qball_regularization
        self.solver_func = qball_regularization
        self.solver_params = solver_params
        return CsaOdfModel.fit(self, *args, **kwargs)

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        sh_coef = CsaOdfModel._get_shm_coef(self, data, mask)
        f = np.dot(sh_coef, self.B.T)
        pd_state, details = self.solver_func(f, self.gtab,
            sampling_matrix=self.B, **self.solver_params)
        self.solver_state = pd_state
        self.solver_details = details
        sh_coef = pd_state[1].T.reshape(sh_coef.shape)
        return sh_coef

class sh_l_tvw_Model(QballBaseModel):
    """ Implementation of Wasserstein-TV model based on HARDI input """
    min = .001
    max = .999
    _n0_const = .5 / np.sqrt(np.pi)

    def fit(self, *args, solver_engine="cvx", solver_params={}, **kwargs):
        if solver_engine == "cvx":
            from qball.solvers.sh_l_tvw.cvx import fit_hardi_qball
        else:
            from qball.solvers.sh_l_tvw.cuda import fit_hardi_qball
        self.solver_func = fit_hardi_qball
        self.solver_params = solver_params
        return QballBaseModel.fit(self, *args, **kwargs)

    def _set_fit_matrix(self, B, L, F, smooth):
        """ The fit matrix describes the forward model. """
        self._fit_matrix = (F * L) / (8 * np.pi)

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        data = data[..., self._where_dwi]
        data = data.clip(self.min, self.max)
        Minv = np.zeros(self._fit_matrix.shape)
        Minv[1:] = 1.0/self._fit_matrix[1:]
        pd_state, details = self.solver_func(data, self.gtab,
            sampling_matrix=self.B, model_matrix=Minv, **self.solver_params)
        self.solver_state = pd_state
        self.solver_details = details
        sh_coef = pd_state[2].T.reshape(data.shape[:-1]+(self.B.shape[1],))
        sh_coef[..., 0] = self._n0_const
        return sh_coef

class sh_l_tvc_Model(QballBaseModel):
    """ Implementation of Ouyang's TV model """
    min = .001
    max = .999
    _n0_const = .5 / np.sqrt(np.pi)

    def fit(self, *args, solver_engine="cvx", solver_params={}, **kwargs):
        if solver_engine == "cvx":
            from qball.solvers.sh_l_tvc.cvx import fit_hardi_qball
        else:
            from qball.solvers.sh_l_tvc.cuda import fit_hardi_qball
        self.solver_func = fit_hardi_qball
        self.solver_params = solver_params
        return QballBaseModel.fit(self, *args, **kwargs)

    def _set_fit_matrix(self, B, L, F, smooth):
        """ The fit matrix describes the forward model. """
        self._fit_matrix = (F * L) / (8 * np.pi)

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        data = data[..., self._where_dwi]
        data = data.clip(self.min, self.max)
        Minv = np.zeros(self._fit_matrix.shape)
        Minv[1:] = 1.0/self._fit_matrix[1:]
        pd_state, details = self.solver_func(data, self.gtab,
            sampling_matrix=self.B, model_matrix=Minv, **self.solver_params)
        self.solver_state = pd_state
        self.solver_details = details
        sh_coef = pd_state[2].T.reshape(data.shape[:-1]+(self.B.shape[1],))
        sh_coef[..., 0] = self._n0_const
        return sh_coef

class sh_l_tvo_Model(QballBaseModel):
    """ Implementation of ODF-TV model """
    min = .001
    max = .999
    _n0_const = .5 / np.sqrt(np.pi)

    def fit(self, *args, solver_engine="cvx", solver_params={}, **kwargs):
        if solver_engine == "cvx":
            from qball.solvers.sh_l_tvo.cvx import fit_hardi_qball
        else:
            from qball.solvers.sh_l_tvo.cuda import fit_hardi_qball
        self.solver_func = fit_hardi_qball
        self.solver_params = solver_params
        return QballBaseModel.fit(self, *args, **kwargs)

    def _set_fit_matrix(self, B, L, F, smooth):
        """ The fit matrix describes the forward model. """
        self._fit_matrix = (F * L) / (8 * np.pi)

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        data = data[..., self._where_dwi]
        data = data.clip(self.min, self.max)
        Minv = np.zeros(self._fit_matrix.shape)
        Minv[1:] = 1.0/self._fit_matrix[1:]
        pd_state, details = self.solver_func(data, self.gtab,
            sampling_matrix=self.B, model_matrix=Minv, **self.solver_params)
        self.solver_state = pd_state
        self.solver_details = details
        sh_coef = pd_state[2].T.reshape(data.shape[:-1]+(self.B.shape[1],))
        sh_coef[..., 0] = self._n0_const
        return sh_coef
