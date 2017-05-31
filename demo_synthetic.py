
import sys
import logging
logging.basicConfig(
    stream=sys.stdout,
    format="[%(relativeCreated) 8d] %(message)s",
    level=logging.DEBUG
)

import numpy as np
import dipy.core.sphere
from dipy.reconst.shm import CsaOdfModel as AganjModel
from dipy.viz import fvtk

from tools import normalize_odf
from model_wtv import WassersteinModel
from model_aganj_wtv import AganjWassersteinModel
from solve_cuda import w1_tv_regularization
import gen

logging.info("Data setup.")
S_data, gtab = gen.synthetic()

l_labels = S_data.shape[-1]
imagedims = S_data.shape[:-1]
d_image = len(imagedims)
n_image = np.prod(imagedims)

assert(gtab.bvals is not None)
assert(gtab.bvecs.shape[1] == 3)
assert(S_data.shape[-1] == gtab.bvals.size)

b_vecs = gtab.bvecs[gtab.bvals > 0,...]
qball_sphere = dipy.core.sphere.Sphere(xyz=b_vecs)

logging.info("Model setup.")
models = [
    AganjModel(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
#    WassersteinModel(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
    AganjWassersteinModel(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
]

logging.info("Model fitting.")
us = [np.zeros(S_data.shape).T for m in range(len(models))]
for (j,m) in enumerate(models):
    logging.info("Model: %s" % type(m).__name__)
    u = m.fit(S_data).odf(qball_sphere)
    u = np.clip(u, 0, np.max(u, -1)[..., None])
    us[j][:] = u.T
#logging.info("Model from SSVM")
#us.append(w1_tv_regularization(us[0], gtab)[0].T)
us.reverse()

logging.info("Plot result. Top to bottom.")
if d_image == 2:
    uniform_odf = np.ones((l_labels,), order='C')/l_labels
    spacing = np.tile(uniform_odf, (1, imagedims[1], 1)).T
    plotdata = np.dstack((
        us[0], spacing, us[1]
    ))
else:
    plotdata = np.dstack(us)
plotdata = plotdata[:,:,:,np.newaxis].transpose(1,2,3,0)

plot_scale = 2.4
plot_norm = True

r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(plotdata, qball_sphere, colormap='jet',
                              norm=plot_norm, scale=plot_scale))
fvtk.show(r, size=(1024, 768))

#logging.info("Store plot.")
#fvtk.record(r, out_path='Aganj.png', size=(1024, 768))