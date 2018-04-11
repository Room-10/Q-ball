
import os, logging
import numpy as np

from dipy.viz import fvtk

import dipy.reconst.dti as dti

def plot_as_odf(imgdata, params, qball_sphere, output_dir):
    for img, name in imgdata:
        for i,s in enumerate(params['records']):
            fname = os.path.join(output_dir, "plot-%s-%d.png" % (name,i))
            r = prepare_odfplot(img, params, qball_sphere, p_slice=s)
            fvtk.snapshot(r, size=(1500,1500), offscreen=True, fname=fname)

def prepare_odfplot(data, params, qball_sphere, p_slice=None):
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
    r_data = fvtk.sphere_funcs(plotdata, qball_sphere, colormap='jet',
                               norm=params['norm'], scale=params['scale'])
    fvtk.add(r, r_data)
    r.set_camera(**camera_params)
    return r

def plot_as_dti(gtab, data, params, qball_sphere, output_dir="."):
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
    fvtk.add(r, fvtk.tensor(tenfit.evals, tenfit.evecs, cfa, qball_sphere))
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