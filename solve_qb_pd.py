
from tools_norm import project_gradients, norms_spectral, norms_nuclear
from tools_diff import gradient, divergence

import numpy as np
from numpy.linalg import norm
from numba import jit

def compute_primal_obj(uk, wk, w0k,
                       ubark, wbark, w0bark,
                       ukp1, wkp1, w0kp1,
                       pk, gk, qk, p0k, g0k,
                       pkp1, gkp1, qkp1, p0kp1, g0kp1,
                       sigma, tau, lbd, theta, dataterm_factor,
                       b_sph, f, constraint_u, uconstrloc,
                       dataterm, avgskips, g_norms):
    pgrad = pkp1.copy()
    ggrad = gkp1.copy()
    qgrad = qkp1.copy()
    p0grad = p0kp1.copy()
    g0grad = g0kp1.copy()

    # pgrad = diag(b) Du - P'B'w
    # ggrad^ij = A^j' w^ij
    # qgrad = b'u - 1
    # p0grad = diag(b) (u-f) - P'B'w0 (W1)
    # g0grad^ij = A^j' w0^ij (W1)
    manifold_op(
        ukp1, wkp1, w0kp1,
        pgrad, ggrad, qgrad, p0grad, g0grad,
        b_sph, f, dataterm, avgskips
    )
    norms_nuclear(ggrad, g_norms)

    if dataterm == "linear":
        # obj_p = <u,s>_b + \sum_ij |A^j' w^ij|
        result = np.einsum('ki,ki->',
            ukp1.reshape(ukp1.shape[0], -1),
            np.einsum('k,ki->ki', b_sph.b, f)
        )
    elif dataterm == "quadratic":
        # obj_p = 0.5*<u-f,u-f>_b + \sum_ij |A^j' w^ij|
        result = np.einsum('k,ki->',
            b_sph.b,
            (0.5*(ukp1 - f)**2).reshape(ukp1.shape[0], -1)
        )
    elif dataterm == "W1":
        # obj_p = \sum_ij |A^j' w0^ij| + \sum_ij |A^j' w^ij|
        # the first part converges to the W^1-distance eventually
        g0_norms = g_norms.copy()
        norms_nuclear(g0grad[:,:,:,np.newaxis], g0_norms)
        result = g0_norms.sum()

    obj_p = result + lbd * g_norms.sum()

    if dataterm == "W1":
        # infeas_p = |diag(b) Du - P'B'w| + |diag(b) (u-f) - P'B'w0|
        #          + |b'u - 1| + |max(0,-u)|
        infeas_p = norm(pgrad.ravel(), ord=np.inf) \
            + norm(p0grad.ravel(), ord=np.inf) \
            + norm(qgrad.ravel(), ord=np.inf) \
            + norm(np.fmax(0.0, -ukp1.ravel()), ord=np.inf)
    else:
        # infeas_p = |diag(b) Du - P'B'w| + |b'u - 1| + |max(0,-u)|
        infeas_p = norm(pgrad.ravel(), ord=np.inf) \
            + norm(qgrad.ravel(), ord=np.inf) \
            + norm(np.fmax(0.0, -ukp1.ravel()), ord=np.inf)

    return obj_p, infeas_p

def compute_dual_obj(uk, wk, w0k,
                     ubark, wbark, w0bark,
                     ukp1, wkp1, w0kp1,
                     pk, gk, qk, p0k, g0k,
                     pkp1, gkp1, qkp1, p0kp1, g0kp1,
                     sigma, tau, lbd, theta, dataterm_factor,
                     b_sph, f, constraint_u, uconstrloc,
                     dataterm, avgskips, g_norms):
    ugrad = ukp1.copy()
    wgrad = wkp1.copy()
    w0grad = w0kp1.copy()

    # ugrad = b q' + diag(b) s + diag(b) D' p (linear)
    # ugrad = b q' - diag(b) f + diag(b) D' p (quadratic)
    # ugrad = b q' + diag(b) p0 + diag(b) D' p (W1)
    # wgrad = Ag - BPp
    # w0grad = Ag0 - BPp0 (W1)
    manifold_op_adjoint(
        ugrad, wgrad, w0grad,
        pkp1, gkp1, qkp1, p0kp1, g0kp1,
        b_sph, f, dataterm, avgskips
    )

    if dataterm == "linear":
        # obj_d = -\sum_i q_i + \sum_i\min_k[ugrad_k^i]
        result = np.sum(np.amin(ugrad[:,np.logical_not(uconstrloc)],axis=0)) \
                + np.sum(np.sum(constraint_u[:,uconstrloc] * ugrad[:,uconstrloc], axis=0))
    elif dataterm == "quadratic":
        # obj_d = -\sum_i q_i + 0.5*b*[f^2 - min(0, q + D'p - f)^2]
        l_labels = ugrad.shape[0]
        result = np.einsum('k,ki->', 0.5*b_sph.b, f.reshape(l_labels, -1)**2) \
                - np.einsum('k,ki->', 0.5/b_sph.b, np.fmin(0.0, ugrad.reshape(l_labels, -1))**2)
    elif dataterm == "W1":
        # obj_d = -\sum_i q_i - <f,p0>_b
        result = -np.einsum('ki,ki->',
            f.reshape(f.shape[0], -1),
            np.einsum('k,ki->ki', b_sph.b, p0kp1)
        )

    obj_d = -np.sum(qkp1)*b_sph.b_precond + result

    if dataterm == "W1":
        # infeas_d = |Ag - BPp| + |Ag0 - BPp0|
        #          + |max(0, |g| - lbd)| + |max(0, |g0| - 1.0)|
        #          + |max(0, -ugrad)|
        g0_norms = g_norms.copy()
        norms_spectral(gkp1, g_norms)
        norms_spectral(g0kp1[:,:,:,np.newaxis], g0_norms)
        infeas_d = norm(wgrad.ravel(), ord=np.inf) \
                + norm(w0grad.ravel(), ord=np.inf) \
                + norm(np.fmax(0.0, g_norms - lbd), ord=np.inf) \
                + norm(np.fmax(0.0, g0_norms - 1.0), ord=np.inf) \
                + norm(np.fmax(0.0, -ugrad.ravel()), ord=np.inf)
    else:
        # infeas_d = |Ag - BPp| + |max(0, |g| - lbd)|
        norms_spectral(gkp1, g_norms)
        infeas_d = norm(wgrad.ravel(), ord=np.inf) \
                + norm(np.fmax(0, g_norms - lbd), ord=np.inf)

    return obj_d, infeas_d

def pd_iteration_step(uk, wk, w0k,
                      ubark, wbark, w0bark,
                      ukp1, wkp1, w0kp1,
                      pk, gk, qk, p0k, g0k,
                      pkp1, gkp1, qkp1, p0kp1, g0kp1,
                      sigma, tau, lbd, theta, dataterm_factor,
                      b_sph, f, constraint_u, uconstrloc,
                      dataterm, avgskips, g_norms):
    # duals
    manifold_op(
        ubark, wbark, w0bark,
        pkp1, gkp1, qkp1, p0kp1, g0kp1,
        b_sph, f, dataterm, avgskips
    )
    pkp1[:] = pk + sigma*pkp1
    gkp1[:] = gk + sigma*gkp1
    qkp1[:] = qk + sigma*qkp1
    p0kp1[:] = p0k + sigma*p0kp1
    g0kp1[:] = g0k + sigma*g0kp1
    project_gradients(gkp1, lbd, g_norms)
    project_gradients(g0kp1[:,:,:,np.newaxis], 1.0, g_norms)
    # update
    pk[:] = pkp1
    gk[:] = gkp1
    qk[:] = qkp1
    p0k[:] = p0kp1
    g0k[:] = g0kp1

    # primals
    manifold_op_adjoint(
        ukp1, wkp1, w0kp1,
        pkp1, gkp1, qkp1, p0kp1, g0kp1,
        b_sph, f, dataterm, avgskips
    )
    np.einsum('k,k...->k...', dataterm_factor, uk - tau*ukp1, out=ukp1)
    ukp1[:] = np.fmax(0.0, ukp1)
    ukp1[:,uconstrloc] = constraint_u[:,uconstrloc]
    wkp1[:] = wk - tau*wkp1
    w0kp1[:] = w0k - tau*w0kp1
    # overrelaxation
    ubark[:] = ukp1 + theta * (ukp1 - uk)
    wbark[:] = wkp1 + theta * (wkp1 - wk)
    w0bark[:] = w0kp1 + theta * (w0kp1 - w0k)
    # update
    uk[:] = ukp1
    wk[:] = wkp1
    w0k[:] = w0kp1

def manifold_op(u, w, w0, pgrad, ggrad, qgrad, p0grad, g0grad,
                b_sph, f, dataterm, avgskips):
    """ Apply the linear operator in the model to (u,w,w0).

    Args:
        u : numpy array of shape (l_labels, imagedims...)
        w : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        w0 : numpy array of shape (n_image, m_gradients, s_manifold)
        pgrad : numpy array of shape (l_labels, d_image, n_image)
        ggrad : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        qgrad : numpy array of shape (n_image)
        p0grad : numpy array of shape (l_labels, n_image)
        g0grad : numpy array of shape (n_image, m_gradients, s_manifold)
        avgskips : output of `staggered_diff_avgskips(imagedims)`
    Returns:
        nothing, the result is written to the given arrays `pgrad`, `ggrad`,
        `qgrad`, `p0grad` and `g0grad`.
    """

    l_labels = u.shape[0]
    imagedims = u.shape[1:]
    n_image, m_gradients, s_manifold, d_image = w.shape
    r_points = b_sph.mdims['r_points']

    pgrad[:] = 0

    # pgrad += diag(b) D u (D is the gradient on a staggered grid)
    gradient(pgrad, u, b_sph.b, avgskips)

    # pgrad_t^i += - P^j' B^j' w_t^ij
    _apply_PB(pgrad, b_sph.P, b_sph.B, w)

    # ggrad^ij += A^j' w^ij
    np.einsum('jlm,ijlt->ijmt', b_sph.A, w, out=ggrad)

    # qgrad += b'u - 1
    np.einsum('i,ij->j', b_sph.b, u.reshape(l_labels, n_image), out=qgrad)
    qgrad -= 1.0
    qgrad *= b_sph.b_precond

    if dataterm == "W1":
        # p0grad = diag(b) (u-f)
        np.einsum('k,ki->ki', b_sph.b, (u-f).reshape(l_labels, -1), out=p0grad)
        # p0grad^i += - P^j' B^j' w0^ij
        _apply_PB0(p0grad, b_sph.P, b_sph.B, w0)
        # g0grad^ij += A^j' w0^ij
        np.einsum('jlm,ijl->ijm', b_sph.A, w0, out=g0grad)

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

def manifold_op_adjoint(ugrad, wgrad, w0grad, p, g, q, p0, g0,
                        b_sph, f, dataterm, avgskips):
    """ Apply the adjoint linear operator in the model to (p,g,q,p0,g0).

    Args:
        ugrad : numpy array of shape (l_labels, imagedims...)
        wgrad : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        w0grad : numpy array of shape (n_image, m_gradients, s_manifold)
        p : numpy array of shape (l_labels, d_image, n_image)
        g : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        q : numpy array of shape (n_image)
        p0 : numpy array of shape (l_labels, n_image)
        g0 : numpy array of shape (n_image, m_gradients, s_manifold)
        avgskips : output of `staggered_diff_avgskips(imagedims)`
    Returns:
        nothing, the result is written to the given arrays `ugrad`, `wgrad` and
        `w0grad`.
    """

    l_labels = ugrad.shape[0]
    imagedims = ugrad.shape[1:]
    n_image, m_gradients, s_manifold, d_image = wgrad.shape
    r_points = b_sph.mdims['r_points']

    # ugrad = b q'
    ugrad_flat = ugrad.reshape(l_labels, -1)
    np.einsum('k,i->ki', b_sph.b, b_sph.b_precond*q, out=ugrad_flat)

    if dataterm == "linear":
        # ugrad += diag(b) f (where f=s)
        ugrad_flat += np.einsum('k,ki->ki', b_sph.b, f)
    elif dataterm == "quadratic":
        # ugrad -= diag(b) f
        ugrad_flat -= np.einsum('k,ki->ki', b_sph.b, f.reshape(l_labels, -1))
    elif dataterm == "W1":
        # ugrad += diag(b) p0
        ugrad_flat += np.einsum('k,ki->ki', b_sph.b, p0)

        # w0grad^ij = A^j g0^ij
        np.einsum('jlm,ijm->ijl', b_sph.A, g0, out=w0grad)

        # w0grad^ij += -B^j P^j p0^i
        w0grad -= np.einsum('jlm,jmi->ijl', b_sph.B, p0[b_sph.P])

    # ugrad += diag(b) D' p (where D' = -div with Dirichlet boundary)
    divergence(p, ugrad, b_sph.b, avgskips)

    # wgrad^ij = A^j g^ij
    np.einsum('jlm,ijmt->ijlt', b_sph.A, g, out=wgrad)

    # wgrad_t^ij += -B^j P^j p_t^i
    wgrad -= np.einsum('jlm,jmti->ijlt', b_sph.B, p[b_sph.P])
