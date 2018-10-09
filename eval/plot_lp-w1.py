
import os, sys, logging, pickle
from repyducible.util import output_dir_create

import numpy as np
import matplotlib.pyplot as plt

from dipy.viz import fvtk
import dipy.core.sphere
from dipy.sims.voxel import multi_tensor_odf

try:
    import qball
except:
    import set_qball_path

from qball.tools.w1dist import w1_dist
from qball.sphere import load_sphere
from qball.tools import normalize_odf
from qball.tools.norm import normalize

def synth_unimodal_odfs(qball_sphere, sphere_vol, directions,
                        cutoff=2*np.pi, const_width=3, tightness=4.9):
    verts = qball_sphere.vertices
    l_labels = verts.shape[0]

    # `const_width` unimodals for each entry of `directions`
    val_base = 1e-6*290
    vals = [tightness*val_base, val_base, val_base]
    vecs = normalize(np.array([
        [np.sin(phi*np.pi/180.0), np.cos(phi*np.pi/180.0), 0]
        for phi in directions
    ]).T).T
    voxels = [
        multi_tensor_odf(verts, np.array((vals, vals)), [v,v], [50, 50])
        for v in vecs
    ]

    if cutoff < 1.5*np.pi:
        # cut off support of distribution functions
        for v,vox in zip(vecs, voxels):
            for k in range(l_labels):
                v_diff1 = verts[k] - v
                v_diff2 = verts[k] + v
                if np.einsum('n,n->', v_diff1, v_diff1) > cutoff**2 \
                   and np.einsum('n,n->', v_diff2, v_diff2) > cutoff**2:
                    vox[k] = 0

    fin = np.zeros((l_labels, const_width*len(voxels)), order='C')
    for i, vox in enumerate(voxels):
        i1 = i*const_width
        i2 = (i+1)*const_width
        fin[:,i1:i2] = np.tile(vox, (const_width, 1)).T
    normalize_odf(fin, sphere_vol)
    return fin

mf = load_sphere(refinement=4)

qball_sphere = dipy.core.sphere.Sphere(
    xyz=mf.v.T, faces=mf.faces.T)

logging.info("Data generation...")

x = list(range(0,185,5))
fin = synth_unimodal_odfs(
    qball_sphere, mf.b, [0,]+x, const_width=1, tightness=30, cutoff=0.15
)

logging.info("Compute/load distances...")

output_dir = "results/plot_lp-w1"
dists_file = os.path.join(output_dir, "dists.npz")
dists_csv_file = os.path.join(output_dir, "dists.csv")
plots_file = os.path.join(output_dir, "lp-w1-result.pdf")
odfplot_file = os.path.join(output_dir, "lp-w1-data.png")
output_dir_create(output_dir)

try:
    npzfile = np.load(open(dists_file, 'rb'))
    l1_dists = npzfile['l1d']
    l2_dists = npzfile['l2d']
    w1_dists = npzfile['w1d']
except:
    u1 = np.tile(fin[:,0:1], fin.shape[1]-1)
    u2 = fin[:,1:]

    logging.info("L1 distances...")
    l1_dists = np.einsum('ki,k->i', np.abs(u1-u2), mf.b)

    logging.info("L2 distances...")
    l2_dists = np.sqrt(np.einsum('ki,k->i', (u1-u2)**2, mf.b))

    logging.info("W1 distances...")
    w1_dists = w1_dist(u1, u2, mf)

    np.savez(open(dists_file, 'wb'),
        l1d=l1_dists, l2d=l2_dists, w1d=w1_dists,
    )

np.savetxt(dists_csv_file,
    np.vstack((x, l1_dists, l2_dists, w1_dists)).T,
    header="angle,l1d,l2d,w1d\n",
    delimiter=",",
    comments="")

logging.info("Plot generation...")

fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(311)
ax1.plot(x, l1_dists)

ax1 = fig.add_subplot(312, sharex=ax1)
ax1.plot(x, l2_dists)

ax2 = fig.add_subplot(313, sharex=ax1)
ax2.plot(x, w1_dists)

fig.tight_layout()
plt.savefig(plots_file)
#plt.show()

l_labels = mf.mdims['l_labels']
uniform_odf = np.ones((l_labels,1), order='C')/l_labels
odfs = (fin[:,0:1],)
for i in range(1,fin.shape[1],2):
    odfs += (uniform_odf, fin[:,i:(i+1)])
plotdata = np.hstack(odfs)[:,:,np.newaxis]
plotdata = plotdata[:,:,:,np.newaxis].transpose(1,2,3,0)

# plot upd and fin as q-ball data sets
r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(plotdata, qball_sphere, colormap='jet',
                              norm=False, scale=11.0))
fvtk.snapshot(r, size=(2000,1000), offscreen=True, fname=odfplot_file)
#fvtk.show(r, size=(1500, 500))
