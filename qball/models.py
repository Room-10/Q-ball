
import numpy as np

import dipy.core.sphere
from dipy.reconst.odf import OdfFit
from dipy.reconst.shm import QballBaseModel, CsaOdfModel, SphHarmFit
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

def parse_data(data, gtab):
    if type(data) is dict:
        # already in extended format
        return data
    imagedims = data.shape[:-1]
    d_image = len(imagedims)
    if d_image == 1:
        dt = data[:,None,None]
        slc = (slice(None),0,0)
    elif d_image == 2:
        dt = data[:,:,None]
        slc = (slice(None),slice(None),0)
    else:
        dt = data
        slc = (slice(None),slice(None),slice(None))
    return { 'raw': dt, 'slice': slc, 'gtab': gtab, }

class n_w_tvw_Model(CsaOdfModel):
    """ Implementation of Wasserstein-TV model from SSVM 2017 """
    def fit(self, data, model_params={},
            solver_engine="pd", solver_params={}, **kwargs):
        data_ext = parse_data(data, self.gtab)
        data = data_ext['raw'][data_ext['slice']]

        from qball.solvers.n_w_tvw.pd import qball_regularization

        if 'sphere' not in model_params:
            b_vecs = self.gtab.bvecs[self.gtab.bvals > 0,...]
            model_params['sphere'] = dipy.core.sphere.Sphere(xyz=b_vecs)
        sphere = model_params['sphere']

        if 'odf' not in model_params:
            if 'csd_response' in model_params:
                csd_response = model_params['csd_response']
                csd_model = ConstrainedSphericalDeconvModel(self.gtab, csd_response)
                odf_fit = csd_model.fit(data)
            else:
                odf_fit = CsaOdfModel.fit(self, data, **kwargs)
            f = odf_fit.odf(sphere)
            model_params['odf'] = np.clip(f, 0, np.max(f, -1)[..., None])

        self.solver_state, self.solver_details = qball_regularization(
            data_ext, model_params, solver_params)

        u = self.solver_state[0]['u']
        l_labels, imagedims = u.shape[0], u.shape[1:]
        u = u.reshape(l_labels, -1).T.reshape(imagedims + (l_labels,))
        return _TrivialOdfFit(u, sphere)

class _TrivialOdfFit(OdfFit):
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

    def fit(self, data, model_params={},
            solver_engine="cvx", solver_params={}, mask=None):
        data_ext = parse_data(data, self.gtab)
        data = data_ext['raw'][data_ext['slice']]

        import importlib
        module_name = "qball.solvers.sh_w_tvw.%s" % (solver_engine,)
        module = importlib.import_module(module_name)
        solver_func = getattr(module, 'qball_regularization')

        if 'csd_response' in model_params:
            csd_response = model_params['csd_response']
            csd_model = ConstrainedSphericalDeconvModel(self.gtab, csd_response)
            odf_fit = csd_model.fit(data)
        else:
            odf_fit = CsaOdfModel.fit(self, data, mask)

        sh_coef = odf_fit._shm_coef
        f = np.dot(sh_coef, self.B.T)
        model_params['odf'] = np.clip(f, 0, np.max(f, -1)[..., None])
        model_params['sampling_matrix'] = self.B
        Minv = np.zeros(self._fit_matrix_fw.shape)
        Minv[1:] = 1.0/self._fit_matrix_fw[1:]
        model_params['model_matrix'] = Minv

        self.solver_state, self.solver_details = solver_func(
            data_ext, model_params, solver_params)

        sh_coef = self.solver_state[0]['v'].T.reshape(sh_coef.shape)
        return SphHarmFit(self, sh_coef, mask)

    def _set_fit_matrix(self, B, L, F, smooth):
        """ The fit matrix describes the forward model. """
        CsaOdfModel._set_fit_matrix(self, B, L, F, smooth)
        self._fit_matrix_fw = (F * L) / (8 * np.pi)

class _SH_HardiQballBaseModel(QballBaseModel):
    """ Base model for our SH-based HARDI-Q-Ball-fitters """
    min = .001
    max = .999
    _n0_const = .5 / np.sqrt(np.pi)

    def fit(self, data, model_params={},
            solver_engine="cvx", solver_params={}, **kwargs):
        self.data_ext = parse_data(data, self.gtab)
        self.model_params = model_params
        self.solver_params = solver_params

        import importlib
        module_name = "qball.solvers.%s.%s" % (self.solver_name, solver_engine)
        module = importlib.import_module(module_name)
        self.solver_func = getattr(module, 'fit_hardi_qball')

        data = self.data_ext['raw'][self.data_ext['slice']]
        return QballBaseModel.fit(self, data, **kwargs)

    def _set_fit_matrix(self, B, L, F, smooth):
        """ The fit matrix describes the forward model. """
        self._fit_matrix = (F * L) / (8 * np.pi)

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        Minv = np.zeros(self._fit_matrix.shape)
        Minv[1:] = 1.0/self._fit_matrix[1:]
        self.model_params['model_matrix'] = Minv
        self.model_params['sampling_matrix'] = self.B
        self.solver_state, self.solver_details = self.solver_func(
            self.data_ext, self.model_params, self.solver_params)
        sh_coef = self.solver_state[0]['v']
        sh_coef = sh_coef.T.reshape(data.shape[:-1]+(self.B.shape[1],))
        sh_coef[..., 0] = self._n0_const
        return sh_coef

sh_hardi_qball_models = [
    "sh_l_tvw",
    "sh_l_tvc",
    "sh_l_tvo",
    "sh_bndl1_tvc",
    "sh_bndl2_tvc",
    "sh_bndl2_tvw",
]

for m in sh_hardi_qball_models:
    mname = "%s_Model" % m
    mcls = type(mname, (_SH_HardiQballBaseModel,), { "solver_name": m })
    globals()[mname] = mcls
