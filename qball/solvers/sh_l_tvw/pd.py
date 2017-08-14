
from qball.tools import apply_PB, apply_PB0
from qball.tools.blocks import BlockVar
from qball.tools.norm import project_gradients, norms_spectral, norms_nuclear
from qball.tools.diff import gradient, divergence
from qball.solvers import PDHGModelHARDI

import numpy as np
from numpy.linalg import norm

import logging

def fit_hardi_qball(data, gtab, sampling_matrix, model_matrix,
                         lbd=1.0, **kwargs):
    solver = MyPDHGModel(data, gtab, sampling_matrix, model_matrix, lbd=lbd)
    details = solver.solve(**kwargs)
    return solver.state, details

class MyPDHGModel(PDHGModelHARDI):
    def __init__(self, data, gtab, sampling_matrix, model_matrix,
                       constraint_u=None, **kwargs):
        PDHGModelHARDI.__init__(self, data, gtab, **kwargs)

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

        c['Y'] = np.zeros(sampling_matrix.shape, order='C')
        c['Y'][:] = sampling_matrix
        l_shm = c['Y'].shape[1]
        c['l_shm'] = l_shm
        c['M'] = model_matrix
        assert(model_matrix.size == l_shm)

        if constraint_u is None:
            c['constraint_u'] = np.zeros((l_labels,) + imagedims, order='C')
            c['constraint_u'][:] = np.nan
        else:
            c['constraint_u'] = constraint_u
        uconstrloc = np.any(np.logical_not(np.isnan(c['constraint_u'])), axis=0)
        c['uconstrloc'] = uconstrloc

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
            resource_stream('qball.solvers.sh_l_tvw', 'cuda_primal.cu'),
            resource_stream('qball.solvers.sh_l_tvw', 'cuda_dual.cu'),
        ]

        e['g_norms'] = np.zeros((n_image, m_gradients), order='C')

        i['xk'] = BlockVar(
            ('u1', (l_labels,) + imagedims),
            ('u2', (l_labels, n_image)),
            ('v', (l_shm, n_image)),
            ('w', (n_image, m_gradients, s_manifold, d_image)),
        )
        i['yk'] = BlockVar(
            ('p', (l_labels, d_image, n_image)),
            ('g', (n_image, m_gradients, s_manifold, d_image)),
            ('q0', (n_image,)),
            ('q1', (l_labels, n_image)),
            ('q2', (l_labels, n_image)),
        )

        # start with a uniform distribution in each voxel
        u1k = i['xk']['u1']
        u1k[:] = 1.0/np.einsum('k->', c['b'])
        u1k[:,c['uconstrloc']] = c['constraint_u'][:,c['uconstrloc']]

        vk = i['xk']['v']
        vk[0,:] = .5 / np.sqrt(np.pi)

        logging.info("HARDI PDHG setup ({l_labels} labels, {l_shm} shm, " \
                     "m={m}; img: {imagedims}; lambda={lbd:.3g}) ready.".format(
                         lbd=c['lbd'],
                         m=m_gradients,
                         l_labels=l_labels,
                         l_shm=l_shm,
                         imagedims="x".join(map(str,imagedims))))

    def obj_primal(self, x, ygrad):
        # ygrad is precomputed via self.linop(x, ygrad)
        #   pgrad = diag(b) Du1 - P'B'w
        #   ggrad^ij = A^j' w^ij
        #   q0grad = b'u1
        #   q1grad = Yv - u1
        #   q2grad = YMv - u2
        u1, u2, v, w = x.vars()
        pgrad, ggrad, q0grad, q1grad, q2grad = ygrad.vars()
        c = self.constvars
        e = self.extravars

        norms_nuclear(ggrad, e['g_norms'])

        # obj_p = 0.5*<u2-f,u2-f>_b + lbd*\sum_ij |A^j' w^ij|
        obj_p = np.einsum('k,ki->', c['b'], 0.5*(u2 - c['f'])**2) \
              + c['lbd']*e['g_norms'].sum()

        # infeas_p = |diag(b) Du1 - P'B'w| + |b'u1 - 1|
        #          + |Yv - u1| + |YMv - u2| + |max(0,-u1)|
        infeas_p = norm(pgrad.ravel(), ord=np.inf) \
            + norm(q0grad.ravel() - c['b_precond'], ord=np.inf) \
            + norm(q1grad.ravel(), ord=np.inf) \
            + norm(q2grad.ravel(), ord=np.inf) \
            + norm(np.fmax(0.0, -u1.ravel()), ord=np.inf)

        return obj_p, infeas_p

    def obj_dual(self, xgrad, y):
        # xgrad is precomputed via self.linop_adjoint(xgrad, y)
        #   u1grad = b q0' - q1 + diag(b) D' p
        #   u2grad = -q2
        #   vgrad = Y'q1 + M Y'q2
        #   wgrad = Ag - BPp
        p, g, q0, q1, q2 = y.vars()
        u1grad, u2grad, vgrad, wgrad = xgrad.vars()
        c = self.constvars
        e = self.extravars
        l_labels = u1grad.shape[0]

        # obj_d = -\sum_i q0_i + 0.5*b*[f^2 - (diag(1/b) q2 + f)^2]
        u2tmp = u2grad - np.einsum('k,ki->ki', c['b'], c['f'])
        obj_d = -np.sum(q0)*c['b_precond'] \
              + np.einsum('k,ki->', 0.5*c['b'], c['f']**2) \
              - np.einsum('k,ki->', 0.5/c['b'], u2tmp**2)

        # infeas_d = |Y'q1 + M Y'q2| + |Ag - BPp| + |max(0, |g| - lbd)|
        norms_spectral(g, e['g_norms'])
        infeas_d = norm(vgrad.ravel(), ord=np.inf) \
                + norm(wgrad.ravel(), ord=np.inf) \
                + norm(np.fmax(0, e['g_norms'] - c['lbd']), ord=np.inf) \
                + norm(np.fmax(0.0, -u1grad.ravel()), ord=np.inf)

        return obj_d, infeas_d

    def linop(self, x, ygrad):
        """ Apply the linear operator in the model to x.

        Args:
            x : primal variable
            ygrad : dual target variable
        Returns:
            nothing, the result is written to the given `ygrad`.
        """
        u1, u2, v, w = x.vars()
        pgrad, ggrad, q0grad, q1grad, q2grad = ygrad.vars()
        c = self.constvars

        l_labels = u1.shape[0]
        imagedims = u1.shape[1:]
        l_shm = v.shape[0]
        n_image, m_gradients, s_manifold, d_image = w.shape

        pgrad[:] = 0

        # pgrad += diag(b) D u1 (D is the gradient on a staggered grid)
        gradient(pgrad, u1, c['b'], c['avgskips'])

        # pgrad_t^i += - P^j' B^j' w_t^ij
        apply_PB(pgrad, c['P'], c['B'], w)

        # ggrad^ij = A^j' w^ij
        np.einsum('jlm,ijlt->ijmt', c['A'], w, out=ggrad)

        # q0grad = b'u1
        np.einsum('i,ij->j', c['b'], u1.reshape(l_labels, n_image), out=q0grad)
        q0grad *= c['b_precond']

        # q1grad = Yv - u1
        np.einsum('km,mi->ki', c['Y'], v, out=q1grad)
        q1grad -= u1.reshape(l_labels, n_image)

        # q2grad = YMv - u2
        np.einsum('km,mi->ki', c['Y'], np.einsum('m,mi->mi', c['M'], v), out=q2grad)
        q2grad -= u2

    def linop_adjoint(self, xgrad, y):
        """ Apply the adjoint linear operator in the model to y

        Args:
            xgrad : primal target variable
            y : dual variable
        Returns:
            nothing, the result is written to the given `xgrad`.
        """
        u1grad, u2grad, vgrad, wgrad = xgrad.vars()
        p, g, q0, q1, q2 = y.vars()
        c = self.constvars

        l_labels = u1grad.shape[0]
        imagedims = u1grad.shape[1:]
        l_shm = vgrad.shape[0]
        n_image, m_gradients, s_manifold, d_image = wgrad.shape

        # u1grad = b q0' - q1
        u1grad_flat = u1grad.reshape(l_labels, -1)
        np.einsum('k,i->ki', c['b'], c['b_precond']*q0, out=u1grad_flat)
        u1grad_flat -= q1

        # u1grad += diag(b) D' p (where D' = -div with Dirichlet boundary)
        divergence(p, u1grad, c['b'], c['avgskips'])

        # u2grad = -q2
        u2grad[:] = -q2

        # vgrad = Y'q1 + M Y'q2
        np.einsum('km,ki->mi', c['Y'], q1, out=vgrad)
        vgrad += np.einsum('m,mi->mi', c['M'], np.einsum('km,ki->mi', c['Y'], q2))

        # wgrad^ij = A^j g^ij
        np.einsum('jlm,ijmt->ijlt', c['A'], g, out=wgrad)

        # wgrad_t^ij += -B^j P^j p_t^i
        wgrad -= np.einsum('jlm,jmti->ijlt', c['B'], p[c['P']])

    def prox_primal(self, x):
        u1 = x['u1']
        u2 = x['u2']
        c = self.constvars

        u2 += c['tau']*np.einsum('k,ki->ki', c['b'], c['f'])
        u2[:] = np.einsum('k,k...->k...', 1.0/(1.0 + c['tau']*c['b']), u2)
        u1[:] = np.fmax(0.0, u1)
        u1[:,c['uconstrloc']] = c['constraint_u'][:,c['uconstrloc']]

    def prox_dual(self, y):
        p, g, q0, q1, q2 = y.vars()
        c = self.constvars
        e = self.extravars

        project_gradients(g, c['lbd'], e['g_norms'])
        q0 -= c['sigma']*c['b_precond']
