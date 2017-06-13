
import sys
import logging
logging.basicConfig(
    stream=sys.stdout,
    format="[%(relativeCreated) 8d] %(message)s",
    level=logging.DEBUG
)
logging.info("Running from command line: %s" % sys.argv)

import numpy as np
import dipy.core.sphere
from dipy.reconst.shm import CsaOdfModel as AganjModel
from dipy.viz import fvtk

from model_wtv import WassersteinModel, WassersteinModelGPU, WassersteinModelCVX
from model_aganj_wtv import AganjWassersteinModel, AganjWassersteinModelGPU, AganjWassersteinModelCVX
from solve_qb_cuda import w1_tv_regularization
import tools_gen as gen

# ==============================================================================
#    Q-ball data preparation
# ==============================================================================

logging.info("Data setup.")
#np.random.seed(seed=234234)
S_data, gtab = gen.synthetic_unimodals()

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
#    AganjWassersteinModelGPU(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
#    AganjWassersteinModel(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
#    AganjWassersteinModelCVX(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
#    WassersteinModelGPU(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
#    WassersteinModel(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
#    WassersteinModelCVX(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
]

logging.info("Model fitting.")
us = [np.zeros(S_data.shape).T for m in range(len(models))]
for (j,m) in enumerate(models):
    logging.info("Model: %s" % type(m).__name__)
    u = m.fit(S_data).odf(qball_sphere)
    u = np.clip(u, 0, np.max(u, -1)[..., None])
    us[j][:] = u.T

logging.info("Model from SSVM")
us.append(np.zeros(S_data.shape).T)
params = {
    'lbd': 0.9,
    'term_relgap': 1e-7,
    'term_maxiter': 80000,
    'granularity': 2000,
    'step_factor': 0.0001,
    'step_bound': 1.3,
    'dataterm': "W1",
    'use_gpu': True
}
pd_state, details = w1_tv_regularization(us[0], gtab, **params)
us[-1][:] = pd_state[0]
us.reverse()

# ==============================================================================
#    Q-ball plot
# ==============================================================================

logging.info("Plot result. Top to bottom:\n%s\nModel from SSVM"
    % "\n".join(type(m).__name__  for m in models))
if d_image == 2:
    uniform_odf = np.ones((l_labels,), order='C')/l_labels
    spacing = np.tile(uniform_odf, (1, imagedims[1], 1)).T
    plotdata = np.dstack((
        us[0], spacing, us[1]
    ))
else:
    plotdata = np.dstack(us)
plotdata = plotdata[:,:,:,np.newaxis].transpose(1,2,3,0)

plot_scale = 1.0
plot_norm = False

r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(plotdata, qball_sphere, colormap='jet',
                              norm=plot_norm, scale=plot_scale))

if len(sys.argv) > 1:
    logging.info("Store plot.")
    r.reset_clipping_range()
    fvtk.snapshot(r, size=(1500,1500), offscreen=True, fname='plot_1d.png')
else:
    fvtk.show(r, size=(768, 1024))
