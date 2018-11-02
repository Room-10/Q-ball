
import logging
import numpy as np

from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sym_sh_basis
from scipy.special import lpn

from repyducible.model import PDBaseModel

from qball.tools import clip_hardi_data

class ModelHARDI(PDBaseModel):
    "Base class for HARDI reconstruction models."
    def __init__(self, *args, lbd=1.0, inpaintloc=None, **kwargs):
        PDBaseModel.__init__(self, *args, **kwargs)

        self.constvars = {}
        b_sph = self.data.b_sph
        gtab = self.data.gtab

        data = self.data.raw[self.data.slice]
        data = np.array(data, dtype=np.float64, copy=True)
        if not self.data.normed:
            data.clip(1.0, out=data)
            b0 = data[...,(gtab.bvals == 0)].mean(-1)
            data /= b0[...,None]
        data = data[...,gtab.bvals > 0]

        imagedims = data.shape[:-1]
        n_image = np.prod(imagedims)
        d_image = len(imagedims)
        l_labels = b_sph.mdims['l_labels']
        s_manifold = b_sph.mdims['s_manifold']
        m_gradients = b_sph.mdims['m_gradients']
        r_points = b_sph.mdims['r_points']
        assert(data.shape[-1] == l_labels)

        c = self.constvars

        c['lbd'] = lbd
        c['b'] = b_sph.b
        c['A'] = b_sph.A
        c['B'] = b_sph.B
        c['P'] = b_sph.P
        c['b_precond'] = b_sph.b_precond
        c['imagedims'] = imagedims
        c['l_labels'] = l_labels
        c['n_image'] = n_image
        c['m_gradients'] = m_gradients
        c['s_manifold'] = s_manifold
        c['d_image'] = d_image
        c['r_points'] = r_points

        inpaintloc = np.zeros(imagedims) if inpaintloc is None else inpaintloc
        c['inpaint_nloc'] = np.ascontiguousarray(np.logical_not(inpaintloc)).ravel()
        assert(c['inpaint_nloc'].shape == (n_image,))

        c['f'] = np.zeros((n_image, l_labels), order='C')
        clip_hardi_data(data)
        loglog_data = np.log(-np.log(data))
        c['f'][:] = loglog_data.reshape(-1, l_labels)
        f_mean = np.einsum('ik,k->i', c['f'], c['b'])/(4*np.pi)
        c['f'] -= f_mean[:,None]

class ModelHARDI_SHM(ModelHARDI):
    "Base class for HARDI reconstr. using spherical harmonics."
    def __init__(self, *args, sh_order=6, smooth=0.006, **kwargs):
        ModelHARDI.__init__(self, *args, **kwargs)

        c = self.constvars

        x, y, z = self.data.gtab.gradients[self.data.gtab.bvals > 0].T
        r, theta, phi = cart2sphere(x, y, z)
        B, m, n = real_sym_sh_basis(sh_order, theta[:, None], phi[:, None])
        L = -n * (n + 1)
        legendre0 = lpn(sh_order, 0)[0]
        F = legendre0[n]
        self.sh_order = sh_order
        self.B = B
        self.m = m
        self.n = n
        self._fit_matrix_fw = (F * L) / (8 * np.pi)

        c['Y'] = np.ascontiguousarray(self.B)
        c['l_shm'] = c['Y'].shape[1]
        c['M'] = np.zeros(self._fit_matrix_fw.shape)
        c['M'][1:] = 1.0/self._fit_matrix_fw[1:]
        assert(c['M'].size == c['l_shm'])
        c['YM'] = np.einsum('lk,k->lk', c['Y'], c['M'])

        logging.info("HARDI SHM setup ({l_labels} labels, {l_shm} shm; " \
                     "img: {imagedims}; lambda={lbd:.3g}) ready.".format(
                         lbd=c['lbd'],
                         l_labels=c['l_labels'],
                         l_shm=c['l_shm'],
                         imagedims="x".join(map(str,c['imagedims']))))

    def pre_cvx(self, data):
        x_raw, y_raw = self.cvx_x.new(), self.cvx_y.new()
        x, y = self.cvx_x.vars(x_raw, True), self.cvx_y.vars(y_raw, True)
        cvx_xvarconv(self.y, data[1], x)
        cvx_yvarconv(self.x, data[0], y)
        return (x_raw, y_raw)

    def post_cvx(self, data):
        c = self.constvars
        x_raw, y_raw = self.x.new(), self.y.new()
        x, y = self.x.vars(x_raw, True), self.y.vars(y_raw, True)
        cvx_yvarconv(self.cvx_y, data[1], x)
        if 'u2' in x:
            np.einsum('km,im->ik', c['Y'],
                np.einsum('m,im->im', c['M'], x['v']), out=x['u2'])
        cvx_xvarconv(self.cvx_x, data[0], y)
        return (x_raw, y_raw)

def cvx_yvarconv(invar, indata, outvar):
    for arr, descr in zip(invar.vars(indata), invar.vars()):
        if descr['name'] not in ["u2","misc"]:
            outvar[descr['name']][:] = arr

def cvx_xvarconv(invar, indata, outvar):
    for arr, descr in zip(invar.vars(indata), invar.vars()):
        if descr['name'] in ["q1","q2","q3","q4","p0"]:
            outvar[descr['name']][:] = arr.transpose((1,0))
        elif descr['name'] in ["g0"]:
            outvar[descr['name']][:] = arr.transpose((1,0,2))
        elif descr['name'] in ["p"]:
            outvar[descr['name']][:] = arr.transpose((2,1,0))
        elif descr['name'] in ["g"]:
            outvar[descr['name']][:] = arr.transpose((1,0,3,2))
        else:
            outvar[descr['name']][:] = arr
