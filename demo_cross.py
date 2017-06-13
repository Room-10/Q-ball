
from __future__ import division

import sys
import logging
logging.basicConfig(
    stream=sys.stdout,
    format="[%(relativeCreated) 8d] %(message)s",
    level=logging.DEBUG
)
logging.info("Running from command line: %s" % sys.argv)

import numpy as np
import matplotlib.pyplot as plt

import dipy.core.sphere
from dipy.viz import fvtk

from dipy.reconst.shm import CsaOdfModel as AganjModel
from model_wtv import WassersteinModelGPU
from model_aganj_wtv import AganjWassersteinModelGPU
from solve_qb_cuda import w1_tv_regularization

from tools_gen import FiberPhantom

# ==============================================================================
#    Fiber phantom preparation
# ==============================================================================

res = 15
f1 = lambda x: 0.5*(x + 0.3)**3 + 0.05
f1inv = lambda y: (y/0.5)**(1/3) - 0.3
f2 = lambda x: 0.7*(1.5 - x)**3 - 0.5
f2inv = lambda y: 1.5 - ((y + 0.5)/0.7)**(1/3)

p = FiberPhantom(res)
p.add_curve(lambda t: (t,f1(t)), tmin=-0.2, tmax=f1inv(1.0)+0.2)
p.add_curve(lambda t: (t,f2(t)), tmin=f2inv(1.0)-0.2, tmax=f2inv(0.0)+0.2)

# ==============================================================================
#    Fiber phantom plot
# ==============================================================================

fig = plt.figure(figsize=(15,5))

subplot_opts = {
    'aspect': 'equal',
    'xticklabels': [],
    'yticklabels': [],
    'xticks': [],
    'yticks': [],
    'xlim': [0.0,1.0],
    'ylim': [0.0,1.0],
}

ax = fig.add_subplot(131, **subplot_opts)
p.plot_curves(ax)
ax = fig.add_subplot(132, **subplot_opts)
p.plot_curves(ax)
p.plot_grid(ax)
ax = fig.add_subplot(133, **subplot_opts)
p.plot_dirs(ax)

plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98,
    wspace=0.03, hspace=0)

if len(sys.argv) > 1:
    plt.savefig("plot_cross_phantom.pdf")
else:
    plt.show()

# ==============================================================================
#    Q-ball data preparation
# ==============================================================================

gtab, S_data = p.gen_hardi(snr=20)

l_labels = S_data.shape[-1]
imagedims = S_data.shape[:-1]
d_image = len(imagedims)
n_image = np.prod(imagedims)

b_vecs = gtab.bvecs[gtab.bvals > 0,...]
qball_sphere = dipy.core.sphere.Sphere(xyz=b_vecs)
models = [
    AganjModel(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
#    AganjWassersteinModelGPU(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
    WassersteinModelGPU(gtab, sh_order=6, smooth=0, min_signal=0, assume_normed=True),
]
logging.info("Model fitting.")
us = [np.zeros((l_labels,) + imagedims, order='C') for m in range(len(models))]
for (j,m) in enumerate(models):
    logging.info("Model: %s" % type(m).__name__)
    u = m.fit(S_data).odf(qball_sphere)
    u = np.clip(u, 0, np.max(u, -1)[..., None])
    us[j][:] = u.T

#logging.info("Model from SSVM")
#us.append(np.zeros(S_data.shape).T)
#params = {
#    'lbd': 0.9,
#    'term_relgap': 1e-05,
#    'term_maxiter': 80000,
#    'granularity': 5000,
#    'step_factor': 0.0001,
#    'step_bound': 1.3,
#    'dataterm': "W1",
#    'use_gpu': True
#}
#pd_state, details = w1_tv_regularization(us[0], gtab, **params)
#us[-1][:] = pd_state[0]
us.reverse()

# ==============================================================================
#    Q-ball plot
# ==============================================================================

logging.info("Plot result. Top to bottom:\n%s\nModel from SSVM"
    % "\n".join(type(m).__name__  for m in models))
if d_image == 2:
    uniform_odf = np.ones((l_labels,), order='C')/l_labels
    spacing = np.tile(uniform_odf, (imagedims[1], 1, 1, 1))
    stack = []
    for i, u in enumerate(us):
        stack.append(u[..., None].transpose(2,1,3,0))
        if i < len(us)-1:
            stack.append(spacing)
    plotdata = np.concatenate(stack, axis=1)
else:
    plotdata = np.dstack(us)[..., None].transpose(1,2,3,0)

plot_scale = 0.8
plot_norm = False

r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(plotdata, qball_sphere, colormap='jet',
                              norm=plot_norm, scale=plot_scale))

if len(sys.argv) > 1:
    logging.info("Store plot.")
    r.reset_clipping_range()
    fvtk.snapshot(r, size=(1500,2000), offscreen=True, fname='plot_cross.png')
else:
    fvtk.show(r, size=(768, 1024))
