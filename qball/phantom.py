
import itertools

import numpy as np
import matplotlib.collections

from dipy.core.gradients import GradientTable
from dipy.sims.voxel import multi_tensor

from qball.sphere import load_sphere

def seg_normal(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    v = p2 - p1
    n = np.array([-v[1], v[0]])
    n /= np.sum(n**2)**0.5
    return n

def translate_segs(segs, delta, crop=None):
    p = np.array(segs[0]) + delta*seg_normal(segs[0], segs[1])
    result = [tuple(p)]
    for i in range(1,len(segs)-1):
        p = np.array(segs[i]) + delta*seg_normal(segs[i-1], segs[i+1])
        result.append(tuple(p))
    p = np.array(segs[-1]) + delta*seg_normal(segs[-2], segs[-1])
    result.append(tuple(p))
    if crop is not None:
        crop_segs(result, crop[0], crop[1])
    return result

def seg_cropped(seg, cropmin, cropmax):
    tol = 1e-6
    p1 = np.array(seg[0])
    p2 = np.array(seg[1])
    v = p2 - p1
    t1 = np.zeros(p1.size)
    t2 = np.ones(p1.size)
    for i in range(p1.size):
        if np.abs(v[i]) < tol:
            if p1[i] < cropmin[i] or p1[i] > cropmax[i]:
                return None
        else:
            t1[i] = (cropmin[i] - p1[i])/v[i]
            t2[i] = (cropmax[i] - p1[i])/v[i]
            if t1[i] > t2[i]:
                t1[i], t2[i] = t2[i], t1[i]
    t1 = np.amax(np.fmin(1.0, np.fmax(0.0, t1)))
    t2 = np.amin(np.fmin(1.0, np.fmax(0.0, t2)))
    if t2-t1 < tol:
        return None
    return (tuple(p1 + t1*v), tuple(p1 + t2*v))

def crop_segs(segs, cropmin, cropmax):
    """ crops segs in place """
    cropmin = np.array(cropmin)
    cropmax = np.array(cropmax)
    i = 0
    while i < len(segs)-1:
        s = seg_cropped((segs[i], segs[i+1]), cropmin, cropmax)
        if s is None:
            if i == 0:
                del segs[0]
            else:
                del segs[i+1]
        else:
            segs[i], segs[i+1] = s
            i += 1

def compute_dirs(lines, res):
    dirs = np.zeros((res,res,2))
    d = 1.0/res
    for l in lines:
        for (x,y) in itertools.product(range(res), repeat=2):
            cropmin = (d*x,d*y)
            cropmax = (d*(x+1),d*(y+1))
            for i in range(len(l)-1):
                s = seg_cropped((l[i],l[i+1]), cropmin, cropmax)
                if s is not None:
                    dirs[x,y,:] -= s[1]
                    dirs[x,y,:] += s[0]
    dir_norm = np.amax(np.sum(dirs**2, axis=2)**0.5)
    dirs *= 1.0/dir_norm
    return dirs

class FiberPhantom(object):
    def __init__(self, res):
        self.res = res
        self.delta = 1.0/res
        self.curves = []

    def add_curve(self, c, tmin=0.0, tmax=1.0, n=20):
        delta_t = (tmax - tmin)/n
        segs = [c(tmin + i*delta_t) for i in range(n+1)]
        lines = [
            translate_segs(segs, 0.0 + 0.03*d, crop=((0.0,0.0),(1.0,1.0))) \
            for d in range(7)
        ]
        dirs = compute_dirs(lines, self.res)
        self.curves.append({
            'segs': segs,
            'lines': lines,
            'dirs': dirs
        })

    def plot_curves(self, ax):
        lines = [l for c in self.curves for l in c['lines']]
        lc = matplotlib.collections.LineCollection(lines)
        ax.add_collection(lc)

    def plot_grid(self, ax):
        gridlines = [ [(self.delta*x,0.0),(self.delta*x,1.0)] for x in range(1,self.res)]
        gridlines += [ [(0.0,self.delta*y),(1.0,self.delta*y)] for y in range(1,self.res)]
        lc = matplotlib.collections.LineCollection(gridlines, colors=[(0.0,0.0,0.0,0.3)])
        ax.add_collection(lc)

    def plot_dirs(self, ax):
        dir_scaling = 0.07
        for c in self.curves:
            dirs = c['dirs']
            for (x,y) in itertools.product(range(self.res), repeat=2):
                mid = self.delta*(np.array([x,y]) + 0.5)
                ax.scatter(mid[0], mid[1], s=3, c='r', linewidths=0)
                if np.sum(dirs[x,y,:]**2)**0.5 > 1e-6:
                    data = np.array([
                        mid[:] - 0.5*dir_scaling*dirs[x,y,:],
                        mid[:] + 0.5*dir_scaling*dirs[x,y,:]
                    ]).T
                    ax.plot(data[0], data[1], 'r')

    def plot_phantom(self, output_file=None):
        import matplotlib.pyplot as plt
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
        self.plot_curves(ax)
        ax = fig.add_subplot(132, **subplot_opts)
        self.plot_curves(ax)
        self.plot_grid(ax)
        ax = fig.add_subplot(133, **subplot_opts)
        self.plot_dirs(ax)

        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98,
            wspace=0.03, hspace=0)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)

    def gen_hardi(self, snr=20):
        bval = 3000
        sph = load_sphere(refinement=2)
        gtab = GradientTable(bval * sph.v.T, b0_threshold=0)
        l_labels = gtab.bvecs.shape[0]
        val_base = 1e-6*300
        S_data = np.zeros((self.res, self.res, l_labels), order='C')
        for (x,y) in itertools.product(range(self.res), repeat=2):
            mid = self.delta*(np.array([x,y]) + 0.5)
            norms = [np.sum(c['dirs'][x,y,:]**2)**0.5 for c in self.curves]
            if sum(norms) < 1e-6:
                mevals = np.array([[val_base, val_base, val_base]])
                sticks = np.array([[1,0,0]])
                fracs = [100]
            else:
                fracs = 100.0*np.array(norms)/sum(norms)
                mevals = np.array([
                    [(1.0+norm*4.0)*val_base, val_base, val_base]
                    for norm in norms
                ])
                sticks = np.array([
                    np.array([c['dirs'][x,y,0], c['dirs'][x,y,1], 0])/norm
                    if norm > 1e-6 else np.array([1,0,0])
                    for c, norm in zip(self.curves, norms)
                ])
            signal, _ = multi_tensor(gtab, mevals,
                S0=1., angles=sticks, fractions=fracs, snr=snr)
            S_data[x,y,:] = signal
        return gtab, S_data
