
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
from dipy.viz import fvtk
from dipy.reconst.shm import CsaOdfModel as AganjModel

from models import SSVMModel, WassersteinModel, AganjWassersteinModel
import tools_gen as gen

# ==============================================================================
#    Q-ball data preparation
# ==============================================================================

logging.info("Data setup.")
#np.random.seed(seed=234234)
S_data, gtab = gen.synth_unimodals()

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
base_params = {
    'sh_order': 6,
    'smooth': 0,
    'min_signal': 0,
    'assume_normed': True
}
models = [
    (AganjModel(gtab, **base_params), {}),
    (SSVMModel(gtab, **base_params), {
        'solver_params': {
            'lbd': 2.5,
            'term_relgap': 1e-05,
            'term_maxiter': 100000,
            'granularity': 5000,
            'step_factor': 0.0001,
            'step_bound': 1.3,
            'dataterm': "W1",
            'use_gpu': True
        },
        'sphere': qball_sphere
    }),
#    (AganjWassersteinModel(gtab, **base_params), {
#        'solver_engine': 'cvx',
#        'solver_params': { 'lbd': 10.0, }
#    }),
#    (AganjWassersteinModel(gtab, **base_params), {
#        'solver_engine': 'pd',
#        'solver_params': {
#            'lbd': 10.0,
#            'term_relgap': 1e-05,
#            'term_maxiter': 20000,
#            'granularity': 5000,
#            'step_factor': 0.001,
#            'step_bound': 0.08,
#            'dataterm': "W1",
#            'use_gpu': True
#        }
#    }),
#    (WassersteinModel(gtab, **base_params), {
#        'solver_engine': 'cvx',
#        'solver_params': { 'lbd': 1.0, }
#    }),
#    (WassersteinModel(gtab, **base_params), {
#        'solver_engine': 'pd',
#        'solver_params': {
#            'lbd': 1.0,
#            'term_relgap': 1e-05,
#            'term_maxiter': 150000,
#            'granularity': 5000,
#            'step_factor': 0.001,
#            'step_bound': 0.0012,
#            'use_gpu': True
#        }
#    }),
]

logging.info("Model fitting.")
us = [np.zeros((l_labels,) + imagedims, order='C') for m in range(len(models))]
for j, (m,fit_params) in enumerate(models):
    logging.info("Model: %s" % type(m).__name__)
    u = m.fit(S_data, **fit_params).odf(qball_sphere)
    u = np.clip(u, 0, np.max(u, -1)[..., None])
    us[j][:] = u.T
us.reverse()

# ==============================================================================
#    Q-ball plot
# ==============================================================================

logging.info("Plot result. Top to bottom:\n%s"
    % "\n".join(type(m[0]).__name__  for m in models))
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
    from util import output_dir_create
    output_dir_create("pic")
    r.reset_clipping_range()
    fvtk.snapshot(r, size=(1500,1500), offscreen=True, fname='pic/plot_1d.png')
else:
    fvtk.show(r, size=(1024, 768))
