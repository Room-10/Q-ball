
import logging
import pickle
from repyducible.data import Data as BaseData

import numpy as np

import dipy
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

from qball.sphere import load_sphere
import qball.tools.gen as gen
from qball.tools.bounds import compute_hardi_bounds

class QBallData(BaseData):
    normed = True
    bounds = None
    bounds_alpha = -1

    def init_spheres(self):
        b_vecs = self.gtab.bvecs[self.gtab.bvals > 0,...]
        self.b_sph = load_sphere(vecs=b_vecs.T)
        self.dipy_sph = dipy.core.sphere.Sphere(xyz=b_vecs)

    def init_bounds(self, alpha, mask=None):
        compute_hardi_bounds(self, alpha=alpha, mask=mask)

    def init_odf(self, params={}, csd=False):
        baseparams = {
            'assume_normed': self.normed,
            'sh_order': 6,
        }
        baseparams.update(params)
        if csd:
            logging.info("Using CSD for ground truth reconstruction.")
            basemodel = ConstrainedSphericalDeconvModel(self.gtab, self.csd_response)
        else:
            basemodel = CsaOdfModel(self.gtab, **baseparams)
        S_data = self.raw[self.slice]
        S_data_orig = self.ground_truth[self.slice]
        f = basemodel.fit(S_data).odf(self.dipy_sph)
        self.odf = np.clip(f, 0, np.max(f, -1)[..., None])
        f = basemodel.fit(S_data_orig).odf(self.dipy_sph)
        self.odf_ground_truth = np.clip(f, 0, np.max(f, -1)[..., None])

    @property
    def csd_response(self):
        resp_file = "cache/resp-%s.pickle" % self.name
        try:
            resp = pickle.load(open(resp_file, 'rb'))
        except:
            logging.info("No cached response function, estimating...")
            resp = gen.csd_response(self.gtab, self.raw)
            pickle.dump(resp, open(resp_file, 'wb'))
        return resp