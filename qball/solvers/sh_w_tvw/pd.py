
from qball.tools.blocks import BlockVar
from qball.tools.norm import project_gradients, norms_spectral, norms_nuclear
from qball.tools.diff import gradient, divergence
from qball.solvers import PDHGModelHARDI

import numpy as np
from numpy.linalg import norm
from numba import jit

import logging

def qball_regularization(f, gtab, sampling_matrix,
                         lbd=10.0, dataterm="W1", **kwargs):
    solver = MyPDHGModel(f, gtab, sampling_matrix, dataterm=dataterm, lbd=lbd)
    details = solver.solve(**kwargs)
    return solver.state, details

class MyPDHGModel(PDHGModelHARDI):
    def __init__(self, f, gtab, sampling_matrix,
                       dataterm="W1", constraint_u=None, **kwargs):
        PDHGModelHARDI.__init__(self, f, gtab, **kwargs)

        c = self.constvars
        e = self.extravars
        i = self.itervars

        imagedims = c['imagedims']
        n_image = c['n_image']
        d_image = c['d_image']
        l_labels = c['l_labels']
        s_manifold = c['s_manifold']
        m_gradients = c['m_gradients']
        r_points = c['r_points']
        b_sph = e['b_sph']

        c['Y'] = np.zeros(sampling_matrix.shape, order='C')
        c['Y'][:] = sampling_matrix
        l_shm = c['Y'].shape[1]
        c['l_shm'] = l_shm

        if constraint_u is None:
            c['constraint_u'] = np.zeros((l_labels,) + imagedims, order='C')
            c['constraint_u'][:] = np.nan
        else:
            c['constraint_u'] = constraint_u
        uconstrloc = np.any(np.logical_not(np.isnan(c['constraint_u'])), axis=0)
        c['uconstrloc'] = uconstrloc

        c['dataterm']= dataterm
        self.gpu_constvars['dataterm']= dataterm[0].upper()

        self.cuda_templates = [
            ("PrimalKernel1", (l_labels, 1, 1), (16, 1, 1)),
            ("PrimalKernel2", (s_manifold*m_gradients, n_image, d_image), (16, 16, 1)),
            ("PrimalKernel3", (n_image, l_labels, 1), (16, 16, 1)),
            ("PrimalKernel4", (n_image, l_shm, 1), (16, 16, 1)),
            ("DualKernel1", (s_manifold*m_gradients, n_image, d_image), (16, 16, 1)),
            ("DualKernel2", (l_labels, n_image, d_image), (16, 16, 1)),
            ("DualKernel3", (n_image, m_gradients, 1), (16, 16, 1)),
        ]
        from pkg_resources import resource_stream
        self.cuda_files = [
            resource_stream('qball.solvers.sh_w_tvw', 'cuda_primal.cu'),
            resource_stream('qball.solvers.sh_w_tvw', 'cuda_dual.cu'),
        ]

        if c['dataterm'] == "quadratic":
            e['dataterm_factor'] = 1.0/(1.0 + tau*b_sph.b)
        elif c['dataterm'] == "W1":
            e['dataterm_factor'] = np.ones((l_labels,))
        else:
            raise Exception("Dataterm '%s' not supported!" % dataterm)

        e['g_norms'] = np.zeros((n_image, m_gradients), order='C')

        i['xk'] = BlockVar(
            ('u', (l_labels,) + imagedims),
            ('v', (l_shm, n_image)),
            ('w', (n_image, m_gradients, s_manifold, d_image)),
            ('w0', (n_image, m_gradients, s_manifold))
        )
        i['yk'] = BlockVar(
            ('p', (l_labels, d_image, n_image)),
            ('g', (n_image, m_gradients, s_manifold, d_image)),
            ('q0', (n_image,)),
            ('q1', (l_labels, n_image)),
            ('p0', (l_labels, n_image)),
            ('g0', (n_image, m_gradients, s_manifold))
        )

        # start with a uniform distribution in each voxel
        uk = i['xk']['u']
        uk[:] = 1.0/np.einsum('k->', c['b'])
        uk[:,c['uconstrloc']] = c['constraint_u'][:,c['uconstrloc']]

        vk = i['xk']['v']
        vk[0,:] = .5 / np.sqrt(np.pi)

        logging.info("HARDI PDHG setup ({l_labels} labels, {l_shm} shm, " \
                     "m={m}; img: {imagedims}; dataterm: {dataterm}, " \
                     "lambda={lbd:.3g}) ready.".format(
                         lbd=c['lbd'],
                         m=m_gradients,
                         l_labels=l_labels,
                         l_shm=l_shm,
                         imagedims="x".join(map(str,imagedims)),
                         dataterm=dataterm))

    def obj_primal(self, x, ygrad):
        # ygrad is precomputed via self.linop(x, ygrad)
        #   pgrad = diag(b) Du - P'B'w
        #   ggrad^ij = A^j' w^ij
        #   q0grad = b'u
        #   q1grad = Yv - u
        #   p0grad = diag(b) u - P'B'w0 (W1)
        #   g0grad^ij = A^j' w0^ij (W1)
        u, v, w, w0 = x.vars()
        pgrad, ggrad, q0grad, q1grad, p0grad, g0grad = ygrad.vars()
        c = self.constvars
        e = self.extravars

        norms_nuclear(ggrad, e['g_norms'])
        if c['dataterm'] == "quadratic":
            # obj_p = 0.5*<u-f,u-f>_b + \sum_ij |A^j' w^ij|
            umf_flat = (0.5*(u - c['f'])**2).reshape(u.shape[0], -1)
            result = np.einsum('k,ki->', c['b'], umf_flat)
        elif c['dataterm'] == "W1":
            # obj_p = \sum_ij |A^j' w0^ij| + \sum_ij |A^j' w^ij|
            # the first part converges to the W^1-distance eventually
            g0_norms = e['g_norms'].copy()
            norms_nuclear(g0grad[:,:,:,np.newaxis], g0_norms)
            result = g0_norms.sum()

        obj_p = result + c['lbd'] * e['g_norms'].sum()

        if c['dataterm'] == "quadratic":
            # infeas_p = |diag(b) Du - P'B'w| + |b'u - 1| + |Yv - u| + |max(0,-u)|
            infeas_p = norm(pgrad.ravel(), ord=np.inf) \
                + norm(c['b_precond']*(q0grad.ravel() - 1.0), ord=np.inf) \
                + norm(q1grad.ravel(), ord=np.inf) \
                + norm(np.fmax(0.0, -u.ravel()), ord=np.inf)
        elif c['dataterm'] == "W1":
            # infeas_p = |diag(b) Du - P'B'w| + |diag(b) (u-f) - P'B'w0|
            #          + |b'u - 1| + |Yv - u| + |max(0,-u)|
            f_flat = c['f'].reshape(c['f'].shape[0], -1)
            p0tmp = p0grad - np.einsum('k,ki->ki', c['b'], f_flat)
            infeas_p = norm(pgrad.ravel(), ord=np.inf) \
                + norm(p0tmp.ravel(), ord=np.inf) \
                + norm(c['b_precond']*(q0grad.ravel() - 1.0), ord=np.inf) \
                + norm(q1grad.ravel(), ord=np.inf) \
                + norm(np.fmax(0.0, -u.ravel()), ord=np.inf)

        return obj_p, infeas_p

    def obj_dual(self, xgrad, y):
        # xgrad is precomputed via self.linop_adjoint(xgrad, y)
        #   ugrad = b q0' - q1 - diag(b) f + diag(b) D' p (quadratic)
        #   ugrad = b q0' - q1 + diag(b) p0 + diag(b) D' p (W1)
        #   vgrad = Y'q1
        #   wgrad = Ag - BPp
        #   w0grad = Ag0 - BPp0 (W1)
        p, g, q0, q1, p0, g0 = y.vars()
        ugrad, vgrad, wgrad, w0grad = xgrad.vars()
        c = self.constvars
        e = self.extravars
        l_labels = ugrad.shape[0]

        if c['dataterm'] == "quadratic":
            # obj_d = -\sum_i q0_i + 0.5*b*[f^2 - min(0, q0 + D'p - f)^2]
            f_flat = c['f'].reshape(l_labels, -1)
            u_flat = ugrad.reshape(l_labels, -1)
            result = np.einsum('k,ki->', 0.5*c['b'], f_flat**2) \
                    - np.einsum('k,ki->', 0.5/c['b'], np.fmin(0.0, u_flat)**2)
        elif c['dataterm'] == "W1":
            # obj_d = -\sum_i q0_i - <f,p0>_b
            f_flat = c['f'].reshape(l_labels, -1)
            result = -np.einsum('ki,ki->', f_flat,
                np.einsum('k,ki->ki', c['b'], p0)
            )

        obj_d = -np.sum(q0)*c['b_precond'] + result

        if c['dataterm'] == "quadratic":
            # infeas_d = |Y'q1| + |Ag - BPp| + |max(0, |g| - lbd)|
            norms_spectral(g, e['g_norms'])
            infeas_d = norm(vgrad.ravel(), ord=np.inf) \
                    + norm(wgrad.ravel(), ord=np.inf) \
                    + norm(np.fmax(0, e['g_norms'] - c['lbd']), ord=np.inf)
        elif c['dataterm'] == "W1":
            # infeas_d = |Y'q1| + |Ag - BPp| + |Ag0 - BPp0|
            #          + |max(0, |g| - lbd)| + |max(0, |g0| - 1.0)|
            #          + |max(0, -ugrad)|
            g0_norms = e['g_norms'].copy()
            norms_spectral(g, e['g_norms'])
            norms_spectral(g0[:,:,:,np.newaxis], g0_norms)
            infeas_d = norm(vgrad.ravel(), ord=np.inf) \
                    + norm(wgrad.ravel(), ord=np.inf) \
                    + norm(w0grad.ravel(), ord=np.inf) \
                    + norm(np.fmax(0.0, e['g_norms'] - c['lbd']), ord=np.inf) \
                    + norm(np.fmax(0.0, g0_norms - 1.0), ord=np.inf) \
                    + norm(np.fmax(0.0, -ugrad.ravel()), ord=np.inf)

        return obj_d, infeas_d

    def linop(self, x, ygrad):
        """ Apply the linear operator in the model to x.

        Args:
            x : primal variable
            ygrad : dual target variable
        Returns:
            nothing, the result is written to the given `ygrad`.
        """
        u, v, w, w0 = x.vars()
        pgrad, ggrad, q0grad, q1grad, p0grad, g0grad = ygrad.vars()
        c = self.constvars

        l_labels = u.shape[0]
        imagedims = u.shape[1:]
        l_shm = v.shape[0]
        n_image, m_gradients, s_manifold, d_image = w.shape

        pgrad[:] = 0

        # pgrad += diag(b) D u (D is the gradient on a staggered grid)
        gradient(pgrad, u, c['b'], c['avgskips'])

        # pgrad_t^i += - P^j' B^j' w_t^ij
        _apply_PB(pgrad, c['P'], c['B'], w)

        # ggrad^ij = A^j' w^ij
        np.einsum('jlm,ijlt->ijmt', c['A'], w, out=ggrad)

        # q0grad = b'u
        np.einsum('i,ij->j', c['b'], u.reshape(l_labels, n_image), out=q0grad)
        q0grad *= c['b_precond']

        # q1grad = Yv - u
        np.einsum('km,mi->ki', c['Y'], v, out=q1grad)
        q1grad -= u.reshape(l_labels, n_image)

        if c['dataterm'] == "W1":
            # p0grad = diag(b) u
            np.einsum('k,ki->ki', c['b'], u.reshape(l_labels, -1), out=p0grad)
            # p0grad^i += - P^j' B^j' w0^ij
            _apply_PB0(p0grad, c['P'], c['B'], w0)
            # g0grad^ij += A^j' w0^ij
            np.einsum('jlm,ijl->ijm', c['A'], w0, out=g0grad)

    def linop_adjoint(self, xgrad, y):
        """ Apply the adjoint linear operator in the model to y

        Args:
            xgrad : primal target variable
            y : dual variable
        Returns:
            nothing, the result is written to the given `xgrad`.
        """
        ugrad, vgrad, wgrad, w0grad = xgrad.vars()
        p, g, q0, q1, p0, g0 = y.vars()
        c = self.constvars

        l_labels = ugrad.shape[0]
        imagedims = ugrad.shape[1:]
        l_shm = vgrad.shape[0]
        n_image, m_gradients, s_manifold, d_image = wgrad.shape

        # ugrad = b q0' - q1
        ugrad_flat = ugrad.reshape(l_labels, -1)
        np.einsum('k,i->ki', c['b'], c['b_precond']*q0, out=ugrad_flat)
        ugrad_flat -= q1

        if c['dataterm'] == "quadratic":
            # ugrad -= diag(b) f
            f_flat = c['f'].reshape(l_labels, -1)
            ugrad_flat -= np.einsum('k,ki->ki', c['b'], f_flat)
        elif c['dataterm'] == "W1":
            # ugrad += diag(b) p0
            ugrad_flat += np.einsum('k,ki->ki', c['b'], p0)

            # w0grad^ij = A^j g0^ij
            np.einsum('jlm,ijm->ijl', c['A'], g0, out=w0grad)

            # w0grad^ij += -B^j P^j p0^i
            w0grad -= np.einsum('jlm,jmi->ijl', c['B'], p0[c['P']])

        # ugrad += diag(b) D' p (where D' = -div with Dirichlet boundary)
        divergence(p, ugrad, c['b'], c['avgskips'])

        # vgrad = Y'q1
        np.einsum('km,ki->mi', c['Y'], q1, out=vgrad)

        # wgrad^ij = A^j g^ij
        np.einsum('jlm,ijmt->ijlt', c['A'], g, out=wgrad)

        # wgrad_t^ij += -B^j P^j p_t^i
        wgrad -= np.einsum('jlm,jmti->ijlt', c['B'], p[c['P']])

    def prox_primal(self, x):
        u = x['u']
        c = self.constvars
        e = self.extravars

        u[:] = np.einsum('k,k...->k...', e['dataterm_factor'], u)
        u[:] = np.fmax(0.0, u)
        u[:,c['uconstrloc']] = c['constraint_u'][:,c['uconstrloc']]

    def prox_dual(self, y):
        p, g, q0, q1, p0, g0 = y.vars()
        c = self.constvars
        e = self.extravars
        f_flat = c['f'].reshape(c['f'].shape[0], -1)

        project_gradients(g, c['lbd'], e['g_norms'])
        q0 -= c['sigma']*c['b_precond']
        p0 -= c['sigma']*np.einsum('k,ki->ki', c['b'], f_flat)
        project_gradients(g0[:,:,:,np.newaxis], 1.0, e['g_norms'])

@jit
def _apply_PB(pgrad, P, B, w):
    """
    # TODO: advanced indexing without creating a copy on the lhs. possible??
    pgrad[b_sph.P] -= np.einsum('jlm,ijlt->jmti', b_sph.B, w)
    """
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            for l in range(w.shape[2]):
                for m in range(B.shape[2]):
                    for t in range(w.shape[3]):
                        pgrad[P[j,m],t,i] -= B[j,l,m] * w[i,j,l,t]

@jit
def _apply_PB0(p0grad, P, B, w0):
    """
    # TODO: advanced indexing without creating a copy on the lhs. possible??
    p0grad[b_sph.P] -= np.einsum('jlm,ijl->jmi', b_sph.B, w0)
    """
    for i in range(w0.shape[0]):
        for j in range(w0.shape[1]):
            for l in range(w0.shape[2]):
                for m in range(B.shape[2]):
                    p0grad[P[j,m],i] -= B[j,l,m] * w0[i,j,l]
