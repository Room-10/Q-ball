
import os, logging
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from dipy.viz import fvtk

import dipy.reconst.dti as dti

def plot_as_odf(imgdata, params, dipy_sph, output_dir):
    for img, name in imgdata:
        for i,s in enumerate(params['records']):
            fname = os.path.join(output_dir, "plot-%s-%d.png" % (name,i))
            r = prepare_odfplot(img, params, dipy_sph, p_slice=s)
            fvtk.snapshot(r, size=(1500,1500), offscreen=True, fname=fname)

def prepare_odfplot(data, params, dipy_sph, p_slice=None):
    p_slice = params['slice'] if p_slice is None else p_slice
    data = (data,) if type(data) is np.ndarray else data

    slicedims = data[0][p_slice].shape[:-1]
    l_labels = data[0].shape[-1]

    camera_params, long_ax, view_ax, stack_ax = \
        prepare_plot_camera(slicedims, datalen=len(data), scale=params['scale'])

    stack = [u[p_slice] for u in data]
    if params['spacing']:
        uniform_odf = np.ones((1,1,1,l_labels), order='C')/(4*np.pi)
        tile_descr = [1,1,1,1]
        tile_descr[long_ax] = slicedims[long_ax]
        spacing = np.tile(uniform_odf, tile_descr)
        for i in reversed(range(1,len(stack))):
            stack.insert(i, spacing)
    if stack_ax == max(long_ax,stack_ax):
        stack = list(reversed(stack))

    plotdata = np.concatenate(stack, axis=stack_ax)
    r = fvtk.ren()
    r_data = fvtk.sphere_funcs(plotdata, dipy_sph, colormap='jet',
                               norm=params['norm'], scale=params['scale'])
    fvtk.add(r, r_data)
    r.set_camera(**camera_params)
    return r

def plot_as_dti(gtab, data, params, dipy_sph, output_dir="."):
    logging.info("Fitting data to DTI model...")
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data[params['slice']])

    FA = dti.fractional_anisotropy(tenfit.evals)
    FA = np.clip(FA, 0, 1)
    RGB = dti.color_fa(FA, tenfit.evecs)
    cfa = RGB
    cfa /= cfa.max()

    logging.info("Recording DTI plot...")
    camera_params, long_ax, view_ax, stack_ax = \
        prepare_plot_camera(data[params['slice']].shape[:-1], scale=2.2)
    r = fvtk.ren()
    fvtk.add(r, fvtk.tensor(tenfit.evals, tenfit.evecs, cfa, dipy_sph))
    r.set_camera(**camera_params)
    fname = os.path.join(output_dir, "plot-dti-0.png")
    fvtk.snapshot(r, size=(1500,1500), offscreen=True, fname=fname)

def prepare_plot_camera(slicedims, datalen=1, scale=1.0):
    axes = [0,1,2]
    long_ax = np.argmax(slicedims)
    axes.remove(long_ax)
    stack_ax = axes[0]
    if slicedims[axes[0]] < slicedims[axes[1]]:
        stack_ax = axes[1]
    axes.remove(stack_ax)
    view_ax = axes[0]

    camera_params = {
        'position': [0,0,0],
        'view_up': [0,0,0]
    }
    camera_params['view_up'][max(long_ax,stack_ax)] = 1
    dist = 2*scale*max(slicedims[long_ax],datalen*(slicedims[stack_ax]+1)-1)
    camera_params['position'][view_ax] = -dist if view_ax == 1 else dist

    return camera_params, long_ax, view_ax, stack_ax

def matrix2brl(arr):
    """Converts a binary matrix to unicode braille points"""
    # counting order of braille points
    brls = 2**np.array([[0,3],
                        [1,4],
                        [2,5],
                        [6,7]])
    brl, W, H = "", 2, 4
    padded = np.pad(arr, ((0,H-arr.shape[0]%H),(0,W-arr.shape[1]%W)), 'constant')
    for i in range(0, padded.shape[0], H):
        for j in range(0, padded.shape[1], W):
            brl += chr(10240+int(np.sum(padded[i:i+H,j:j+W]*brls)))
        brl += "\n"
    return brl.strip("\n")

def plot_mesh3(ax, vecs, tris):
    """ Plots a surface according to a given triangulation.

    Args:
        ax : Instance of mpl_toolkits.mplot3d.Axes3d
        vecs : numpy array of shape (3, l_vecs) containing the grid points of the
               surface
        tris : numpy array of shape (3, l_vecs). Each column contains the three
               indices (wrt. `vecs`) of a triangle's vertices.

    Test code:
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure(facecolor="white")
    >>> fig.subplots_adjust(left=0, right=1, top=1, bottom=0.1)
    >>> ax = fig.add_subplot(1, 1, 1, projection='3d')
    >>> ax.set_xlim((-1.1,1.1))
    >>> ax.set_ylim((-1.1,1.1))
    >>> ax.set_zlim((-1.1,1.1))
    >>> ax.view_init(azim=30)
    >>> ax.scatter(1.01*sv.points[:,0], 1.01*sv.points[:,1], 1.01*sv.points[:,2])
    >>> plot_mesh3(ax, sv.points.T, sv._tri.simplices.T)
    >>> plt.show()
    """
    vx = vecs[0]
    vy = vecs[1]
    vz = vecs[2]

    tmp = np.tile(tris, (2,1))

    verts = np.dstack((vx[tmp].T, vy[tmp].T, vz[tmp].T))
    collection = Poly3DCollection(verts, linewidths=0.3, alpha=0.8)
    collection.set_facecolor([1,1,1])
    collection.set_edgecolor([0.5,0.5,0.5])
    ax.add_collection3d(collection)
    #ax.add_collection3d(
    #    Line3DCollection(verts, colors='k', linewidths=0.2, linestyle=':')
    #)
    #for k in range(vecs.shape[1]):
    #    ax.text(1.1*vecs[0,k], 1.1*vecs[1,k], 1.1*vecs[2,k], str(k))
