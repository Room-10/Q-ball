
from qball.tools import normalize_odf, apply_PB
from qball.tools.blocks import BlockVar
from qball.tools.norm import project_gradients, norms_spectral, norms_nuclear
from qball.tools.diff import gradient, divergence
from qball.solvers import PDHGModelHARDI

import numpy as np
from numpy.linalg import norm

import logging

def qball_regularization(f, gtab, lbd=1.0, dataterm="W1", gradnorm="frobenius",
                                  constraint_u=None, inpaintloc=None, **kwargs):
    solver = MyPDHGModel(f, gtab, dataterm=dataterm, gradnorm=gradnorm, lbd=lbd,
                                  constraint_u=constraint_u, inpaintloc=inpaintloc)
    details = solver.solve(**kwargs)
    return solver.state, details

class MyPDHGModel(PDHGModelHARDI):
    def __init__(self, f, gtab, dataterm="W1", gradnorm="frobenius",
                                constraint_u=None, inpaintloc=None, **kwargs):
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

        f_flat = f.reshape(-1, l_labels).T
        c['f'] = np.array(f_flat.reshape((l_labels,) + imagedims), order='C')
        normalize_odf(c['f'], c['b'])

        if dataterm == "linear":
            """ Computes the cost vector from the reference image. """
            s = np.zeros((l_labels, n_image), order='C')
            f_flat = c['f'].reshape(l_labels, n_image)
            b_sph = e['b_sph']
            for k1 in range(l_labels):
                # s_ki = \sum_l 0.5 * b^l * f_i^l * dist(z^l, z^k)^2
                for k2 in range(l_labels):
                    s[k1,:] += c['b'][k2]*f_flat[k2,:]*0.5*\
                        b_sph.dist(b_sph.v[:,k2], b_sph.v[:,k1])**2
            c['f'] = s
        elif dataterm == "linear-precalc":
            """ Nothing to do, the provided f is the cost vector. """
            dataterm = "linear"
        elif dataterm not in ["W1","quadratic"]:
            raise Exception("Dateterm unknown: %s" % dataterm)

        if constraint_u is None:
            c['constraint_u'] = np.zeros((l_labels,) + imagedims, order='C')
            c['constraint_u'][:] = np.nan
        else:
            c['constraint_u'] = constraint_u
        uconstrloc = np.any(np.logical_not(np.isnan(c['constraint_u'])), axis=0)
        c['uconstrloc'] = uconstrloc

        if inpaintloc is None:
            inpaintloc = np.zeros(imagedims)
        c['inpaint_nloc'] = np.ascontiguousarray(np.logical_not(inpaintloc)).ravel()
        assert(c['inpaint_nloc'].shape == (n_image,))

        c['dataterm']= dataterm
        self.gpu_constvars['dataterm']= dataterm[0].upper()

        c['gradnorm']= gradnorm
        self.gpu_constvars['gradnorm']= gradnorm[0].upper()

        e['g_norms'] = np.zeros((n_image, m_gradients), order='C')

        i['xk'] = BlockVar(
            ('u', (l_labels,) + imagedims),
            ('w', (n_image, m_gradients, s_manifold, d_image)),
            ('w0', (n_image, m_gradients, s_manifold))
        )
        i['yk'] = BlockVar(
            ('p', (l_labels, d_image, n_image)),
            ('g', (n_image, m_gradients, s_manifold, d_image)),
            ('q', (n_image,)),
            ('p0', (l_labels, n_image)),
            ('g0', (n_image, m_gradients, s_manifold))
        )

        # start with a uniform distribution in each voxel
        uk = i['xk']['u']
        uk[:] = 1.0/np.einsum('k->', c['b'])
        uk[:,c['uconstrloc']] = c['constraint_u'][:,c['uconstrloc']]

        logging.info("HARDI PDHG setup ({l_labels} labels, m={m}; " \
                     "img: {imagedims}; dataterm: {dataterm}, " \
                     "lambda={lbd:.3g}) ready.".format(
                         lbd=c['lbd'],
                         m=m_gradients,
                         l_labels=l_labels,
                         imagedims="x".join(map(str,imagedims)),
                         dataterm=dataterm))

    def prepare_gpu(self):
        c = self.constvars
        n_image = c['n_image']
        d_image = c['d_image']
        l_labels = c['l_labels']
        s_manifold = c['s_manifold']
        m_gradients = c['m_gradients']

        prox_sg= "PP" if 'precond' in c else "Pd"
        self.cuda_templates = [
            ("prox_primal", prox_sg, (n_image, l_labels, 1), (16, 16, 1)),
            ("prox_dual1", prox_sg, (n_image, l_labels, 1), (16, 16, 1)),
            ("prox_dual2", prox_sg, (n_image, m_gradients, 1), (16, 16, 1)),
            ("linop1", "PP", (s_manifold*m_gradients, n_image, d_image), (16, 16, 1)),
            ("linop2", "PP", (l_labels, n_image, d_image), (16, 16, 1)),
            ("linop_adjoint1", "PP", (l_labels, 1, 1), (512, 1, 1)),
            ("linop_adjoint2", "PP", (s_manifold*m_gradients, n_image, d_image), (16, 16, 1)),
            ("linop_adjoint3", "PP", (n_image, l_labels, 1), (16, 16, 1)),
        ]

        from pkg_resources import resource_stream
        self.cuda_files = [
            resource_stream('qball.solvers.n_w_tvw', 'cuda_primal.cu'),
            resource_stream('qball.solvers.n_w_tvw', 'cuda_dual.cu'),
        ]

        PDHGModelHARDI.prepare_gpu(self)

        def gpu_kernels_linop(*args):
            self.cuda_kernels['linop1'](*args)
            self.cuda_kernels['linop2'](*args)
        self.gpu_kernels['linop'] = gpu_kernels_linop

        def gpu_kernels_linop_adjoint(*args):
            self.cuda_kernels['linop_adjoint1'](*args)
            self.cuda_kernels['linop_adjoint2'](*args)
            self.cuda_kernels['linop_adjoint3'](*args)
        self.gpu_kernels['linop_adjoint'] = gpu_kernels_linop_adjoint

        self.gpu_kernels['prox_primal'] = self.cuda_kernels['prox_primal']
        def gpu_kernels_prox_dual(*args):
            self.cuda_kernels['prox_dual1'](*args)
            self.cuda_kernels['prox_dual2'](*args)
        self.gpu_kernels['prox_dual'] = gpu_kernels_prox_dual

    def obj_primal(self, x, ygrad):
        # ygrad is precomputed via self.linop(x, ygrad)
        #   pgrad = diag(b) Du - P'B'w
        #   ggrad^ij = A^j' w^ij
        #   qgrad = b'u
        #   p0grad = diag(b) u - P'B'w0 (W1)
        #   g0grad^ij = A^j' w0^ij (W1)
        u, w, w0 = x.vars()
        pgrad, ggrad, qgrad, p0grad, g0grad = ygrad.vars()
        c = self.constvars
        e = self.extravars

        norms_nuclear(ggrad, e['g_norms'], c['gradnorm'])
        if c['dataterm'] == "linear":
            # obj_p = <u,s>_b + \sum_ij |A^j' w^ij|
            u_flat = u.reshape(u.shape[0], -1)[:,c['inpaint_nloc']]
            f_flat = c['f'][:,c['inpaint_nloc']]
            result = np.einsum('ki,ki->', u_flat,
                                        np.einsum('k,ki->ki', c['b'], f_flat))
        elif c['dataterm'] == "quadratic":
            # obj_p = 0.5*<u-f,u-f>_b + \sum_ij |A^j' w^ij|
            umf_flat = (0.5*(u - c['f'])**2).reshape(u.shape[0], -1)
            result = np.einsum('k,ki->', c['b'], umf_flat[:,c['inpaint_nloc']])
        elif c['dataterm'] == "W1":
            # obj_p = \sum_ij |A^j' w0^ij| + \sum_ij |A^j' w^ij|
            # the first part converges to the W^1-distance eventually
            g0_norms = e['g_norms'].copy()
            norms_nuclear(g0grad[:,:,:,np.newaxis], g0_norms)
            result = g0_norms[c['inpaint_nloc'],:].sum()

        obj_p = result + c['lbd'] * e['g_norms'].sum()

        # infeas_p = |diag(b) Du - P'B'w| + |b'u - 1| + |max(0,-u)|
        infeas_p = norm(pgrad.ravel(), ord=np.inf) \
                 + norm(qgrad.ravel() - c['b_precond'], ord=np.inf) \
                 + norm(np.fmax(0.0, -u.ravel()), ord=np.inf)
        if c['dataterm'] == "W1":
            # infeas_p += |diag(b) (u-f) - P'B'w0|
            f_flat = c['f'].reshape(c['f'].shape[0], -1)[:,c['inpaint_nloc']]
            p0tmp = p0grad[:,c['inpaint_nloc']] - \
                np.einsum('k,ki->ki', c['b'], f_flat)
            infeas_p = norm(p0tmp.ravel(), ord=np.inf)

        return obj_p, infeas_p

    def obj_dual(self, xgrad, y):
        # xgrad is precomputed via self.linop_adjoint(xgrad, y)
        #   ugrad = b q' + diag(b) D' p (linear, quadratic)
        #   ugrad = b q' + diag(b) D' p + diag(b) p0 (W1)
        #   wgrad = Ag - BPp
        #   w0grad = Ag0 - BPp0 (W1)
        p, g, q, p0, g0 = y.vars()
        ugrad, wgrad, w0grad = xgrad.vars()
        c = self.constvars
        e = self.extravars
        l_labels = ugrad.shape[0]

        if c['dataterm'] == "linear":
            # obj_d = -\sum_i q_i
            result = 0.0
        elif c['dataterm'] == "quadratic":
            # obj_d = -\sum_i q_i + 0.5*b*[f^2 - min(0, q + D'p - f)^2]
            f_flat = c['f'].reshape(l_labels, -1)[:,c['inpaint_nloc']]
            u_flat = ugrad.reshape(l_labels, -1)[:,c['inpaint_nloc']]
            umf_flat = u_flat - np.einsum('k,ki->ki', c['b'], f_flat)
            result = np.einsum('k,ki->', 0.5*c['b'], f_flat**2) \
                    - np.einsum('k,ki->', 0.5/c['b'], np.fmin(0.0, umf_flat)**2)
        elif c['dataterm'] == "W1":
            # obj_d = -\sum_i q_i - <f,p0>_b
            f_flat = c['f'].reshape(l_labels, -1)
            result = -np.einsum('ki,ki->', f_flat[:,c['inpaint_nloc']],
                np.einsum('k,ki->ki', c['b'], p0[:,c['inpaint_nloc']]))

        obj_d = -np.sum(q)*c['b_precond'] + result

        norms_spectral(g, e['g_norms'], c['gradnorm'])
        # infeas_d = |Ag - BPp| + |max(0, |g| - lbd)|
        infeas_d = norm(wgrad.ravel(), ord=np.inf) \
                 + norm(np.fmax(0.0, e['g_norms'] - c['lbd']), ord=np.inf)
        if c['dataterm'] == "W1":
            # infeas_d += |Ag0 - BPp0| + |max(0, |g0| - 1.0)| + |max(0, -ugrad)|
            g0_norms = e['g_norms'].copy()
            norms_spectral(g0[:,:,:,np.newaxis], g0_norms)
            infeas_d += norm(w0grad.ravel(), ord=np.inf) \
                     + norm(np.fmax(0.0, g0_norms[c['inpaint_nloc'],:] - 1.0), ord=np.inf) \
                     + norm(np.fmax(0.0, -ugrad.ravel()), ord=np.inf)
        elif c['dataterm'] == "linear":
            # infeas_d += |max(0, -ugrad - diag(b) s)|
            diagb_s = np.einsum('k,ki->ki', c['b'], c['f'][:,c['inpaint_nloc']]).ravel()
            infeas_d += norm(np.fmax(0.0, -ugrad[:,c['inpaint_nloc']].ravel() - diagb_s), ord=np.inf)

        return obj_d, infeas_d

    def precond(self, x, y):
        u, w, w0 = x.vars()
        p, g, q, p0, g0 = y.vars()
        u_flat = u.reshape(u.shape[0], -1)
        c = self.constvars
        x[:] = 0.0
        y[:] = 0.0

        # p += diag(b) D u (D is the gradient on a staggered grid)
        # p_t^i += -P^j' B^j' w_t^ij
        # g^ij = A^j' w^ij
        # q = b'u
        # p0 = diag(b) u (W1)
        # p0^i += - P^j' B^j' w0^ij (W1)
        # g0^ij += A^j' w0^ij (W1)
        gradient(p, u, c['b'], c['avgskips'], precond=True)
        apply_PB(p, c['P'], c['B'], w, precond=True)
        g += norm(c['A'], ord=1, axis=1)[None,:,:,None]
        q += c['b_precond']*norm(c['b'], ord=1)
        if c['dataterm'] == "W1":
            p0 += np.abs(c['b'])[:,None]
            apply_PB(p0[:,None,:], c['P'], c['B'], w0[:,:,:,None],
                     precond=True, inpaint_nloc=c['inpaint_nloc'])
            g0 += norm(c['A'], ord=1, axis=1)[None,:,:]
        y[y.data > np.spacing(1)] = 1.0/y[y.data > np.spacing(1)]

        # u = b q'
        # u += diag(b) p0 (W1)
        # w0^ij = A^j g0^ij - B^j P^j p0^i (W1)
        # u += diag(b) D' p (where D' = -div with Dirichlet boundary)
        # v = Y'q1
        # w^ij = A^j g^ij - B^j P^j p_t^i
        u_flat += c['b_precond']*np.abs(c['b'])[:,None]
        if c['dataterm'] == "W1":
            u_flat[:,c['inpaint_nloc']] += np.abs(c['b'])[:,None]
            w0[c['inpaint_nloc'],:,:] += norm(c['A'], ord=1, axis=2)[None,:,:]
            w0[c['inpaint_nloc'],:,:] += norm(c['B'], ord=1, axis=2)[None,:,:]
        divergence(p, u, c['b'], c['avgskips'], precond=True)
        w += norm(c['A'], ord=1, axis=2)[None,:,:,None]
        w += norm(c['B'], ord=1, axis=2)[None,:,:,None]
        x[x.data > np.spacing(1)] = 1.0/x[x.data > np.spacing(1)]

    def linop(self, x, ygrad):
        """ Apply the linear operator in the model to x.

        Args:
            x : primal variable
            ygrad : dual target variable
        Returns:
            nothing, the result is written to the given `ygrad`.
        """
        u, w, w0 = x.vars()
        pgrad, ggrad, qgrad, p0grad, g0grad = ygrad.vars()
        c = self.constvars

        l_labels = u.shape[0]
        imagedims = u.shape[1:]
        n_image, m_gradients, s_manifold, d_image = w.shape

        pgrad[:] = 0

        # pgrad += diag(b) D u (D is the gradient on a staggered grid)
        gradient(pgrad, u, c['b'], c['avgskips'])

        # pgrad_t^i += - P^j' B^j' w_t^ij
        apply_PB(pgrad, c['P'], c['B'], w)

        # ggrad^ij = A^j' w^ij
        np.einsum('jlm,ijlt->ijmt', c['A'], w, out=ggrad)

        # qgrad = b'u
        np.einsum('k,ki->i', c['b'], u.reshape(l_labels, n_image), out=qgrad)
        qgrad *= c['b_precond']

        if c['dataterm'] == "W1":
            # p0grad = diag(b) u
            p0grad[:,c['inpaint_nloc']] = np.einsum('k,ki->ki', c['b'],
                u.reshape(l_labels, -1)[:,c['inpaint_nloc']])

            # p0grad^i += - P^j' B^j' w0^ij
            apply_PB(p0grad[:,None,:], c['P'], c['B'], w0[:,:,:,None],
                     inpaint_nloc=c['inpaint_nloc'])

            # g0grad^ij += A^j' w0^ij
            g0grad[c['inpaint_nloc'],:,:] \
                = np.einsum('jlm,ijl->ijm', c['A'], w0[c['inpaint_nloc'],:,:])

    def linop_adjoint(self, xgrad, y):
        """ Apply the adjoint linear operator in the model to y

        Args:
            xgrad : primal target variable
            y : dual variable
        Returns:
            nothing, the result is written to the given `xgrad`.
        """
        ugrad, wgrad, w0grad = xgrad.vars()
        p, g, q, p0, g0 = y.vars()
        c = self.constvars

        l_labels = ugrad.shape[0]
        imagedims = ugrad.shape[1:]
        n_image, m_gradients, s_manifold, d_image = wgrad.shape

        # ugrad = b q'
        ugrad_flat = ugrad.reshape(l_labels, -1)
        np.einsum('k,i->ki', c['b'], c['b_precond']*q, out=ugrad_flat)

        if c['dataterm'] == "W1":
            # ugrad += diag(b) p0
            ugrad_flat[:,c['inpaint_nloc']] \
                += np.einsum('k,ki->ki', c['b'], p0[:,c['inpaint_nloc']])

            # w0grad^ij = A^j g0^ij
            w0grad[c['inpaint_nloc'],:,:] \
                = np.einsum('jlm,ijm->ijl', c['A'], g0[c['inpaint_nloc'],:,:])

            # w0grad^ij += -B^j P^j p0^i
            w0grad[c['inpaint_nloc'],:,:] -= \
                np.einsum('jlm,jmi->ijl', c['B'], p0[:,c['inpaint_nloc']][c['P']])

        # ugrad += diag(b) D' p (where D' = -div with Dirichlet boundary)
        divergence(p, ugrad, c['b'], c['avgskips'])

        # wgrad^ij = A^j g^ij
        np.einsum('jlm,ijmt->ijlt', c['A'], g, out=wgrad)

        # wgrad_t^ij += -B^j P^j p_t^i
        wgrad -= np.einsum('jlm,jmti->ijlt', c['B'], p[c['P']])

    def prox_primal(self, x, tau):
        u = x['u']
        c = self.constvars

        if c['dataterm'] == "linear":
            l_labels = c['l_labels']
            u_flat = u.reshape(l_labels, -1)
            f_flat = c['f'][:,c['inpaint_nloc']]
            utau = tau['u'].reshape(l_labels, -1)[:,c['inpaint_nloc']] \
                   if 'precond' in c else tau
            u_flat[:,c['inpaint_nloc']] -= utau*np.einsum('k,ki->ki', c['b'], f_flat)
        elif c['dataterm'] == "quadratic":
            l_labels = c['l_labels']
            u_flat = u.reshape(l_labels, -1)
            f_flat = c['f'].reshape(l_labels, -1)[:,c['inpaint_nloc']]
            utau = tau['u'].reshape(l_labels, -1)[:,c['inpaint_nloc']] \
                   if 'precond' in c else tau
            u_flat[:,c['inpaint_nloc']] += utau*np.einsum('k,ki->ki', c['b'], f_flat)
            u_flat[:,c['inpaint_nloc']] *= 1.0/(1.0 + utau*c['b'][:,None])

        u[:] = np.fmax(0.0, u)
        u[:,c['uconstrloc']] = c['constraint_u'][:,c['uconstrloc']]

    def prox_dual(self, y, sigma):
        p, g, q, p0, g0 = y.vars()
        c = self.constvars
        e = self.extravars

        project_gradients(g, c['lbd'], e['g_norms'], c['gradnorm'])

        qsigma = sigma['q'] if 'precond' in c else sigma
        q -= qsigma*c['b_precond']

        if c['dataterm'] == "W1":
            project_gradients(g0[:,:,:,np.newaxis], 1.0, e['g_norms'])

            f_flat = c['f'].reshape(c['f'].shape[0], -1)[:,c['inpaint_nloc']]
            p0sigma = sigma['p0'][:,c['inpaint_nloc']] if 'precond' in c else sigma
            p0[:,c['inpaint_nloc']] -= p0sigma*np.einsum('k,ki->ki', c['b'], f_flat)
