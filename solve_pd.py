
from tools_norm import project_gradients, norms_spectral, norms_nuclear
from tools_diff import gradient, divergence

import numpy as np
from numpy.linalg import norm
from numba import jit

def pd_iteration_step(uk, vk, wk, w0k,
                      ubark, vbark, wbark, w0bark,
                      ukp1, vkp1, wkp1, w0kp1,
                      pk, gk, q0k, q1k, p0k, g0k,
                      pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1,
                      sigma, tau, lbd, theta, dataterm_factor,
                      b_sph, f, Y, constraint_u, uconstrloc,
                      dataterm, avgskips, g_norms):
    # duals
    manifold_op(
        ubark, vbark, wbark, w0bark,
        pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1,
        b_sph, f, Y, dataterm, avgskips
    )
    pkp1[:] = pk + sigma*pkp1
    gkp1[:] = gk + sigma*gkp1
    q0kp1[:] = q0k + sigma*q0kp1
    q1kp1[:] = q1k + sigma*q1kp1
    p0kp1[:] = p0k + sigma*p0kp1
    g0kp1[:] = g0k + sigma*g0kp1
    project_gradients(gkp1, lbd, g_norms)
    project_gradients(g0kp1[:,:,:,np.newaxis], 1.0, g_norms)
    # update
    pk[:] = pkp1
    gk[:] = gkp1
    q0k[:] = q0kp1
    q1k[:] = q1kp1
    p0k[:] = p0kp1
    g0k[:] = g0kp1

    # primals
    manifold_op_adjoint(
        ukp1, vkp1, wkp1, w0kp1,
        pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1,
        b_sph, f, Y, dataterm, avgskips
    )
    np.einsum('k,k...->k...', dataterm_factor, uk - tau*ukp1, out=ukp1)
    ukp1[:] = np.fmax(0.0, ukp1)
    ukp1[:,uconstrloc] = constraint_u[:,uconstrloc]
    vkp1[:] = vk - tau*vkp1
    wkp1[:] = wk - tau*wkp1
    w0kp1[:] = w0k - tau*w0kp1
    # overrelaxation
    ubark[:] = ukp1 + theta * (ukp1 - uk)
    vbark[:] = vkp1 + theta * (vkp1 - vk)
    wbark[:] = wkp1 + theta * (wkp1 - wk)
    w0bark[:] = w0kp1 + theta * (w0kp1 - w0k)
    # update
    uk[:] = ukp1
    vk[:] = vkp1
    wk[:] = wkp1
    w0k[:] = w0kp1

def compute_primal_obj(ukp1, vkp1, wkp1, w0kp1,
                       pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1,
                       lbd, f, Y, b_sph, constraint_u, uconstrloc,
                       dataterm, avgskips, g_norms):
    pgrad = pkp1.copy()
    ggrad = gkp1.copy()
    q0grad = q0kp1.copy()
    q1grad = q1kp1.copy()
    p0grad = p0kp1.copy()
    g0grad = g0kp1.copy()

    # pgrad = diag(b) Du - P'B'w
    # ggrad^ij = A^j' w^ij
    # q0grad = b'u - 1
    # q1grad = Yv - u
    # p0grad = diag(b) (u-f) - P'B'w0 (W1)
    # g0grad^ij = A^j' w0^ij (W1)
    manifold_op(
        ukp1, vkp1, wkp1, w0kp1,
        pgrad, ggrad, q0grad, q1grad, p0grad, g0grad,
        b_sph, f, Y, dataterm, avgskips
    )
    norms_nuclear(ggrad, g_norms)

    if dataterm == "quadratic":
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
        #          + |b'u - 1| + |Yv - u| + |max(0,-u)|
        infeas_p = norm(pgrad.ravel(), ord=np.inf) \
            + norm(p0grad.ravel(), ord=np.inf) \
            + norm(q0grad.ravel(), ord=np.inf) \
            + norm(q1grad.ravel(), ord=np.inf) \
            + norm(np.fmax(0.0, -ukp1.ravel()), ord=np.inf)
    else:
        # infeas_p = |diag(b) Du - P'B'w| + |b'u - 1| + |Yv - u| + |max(0,-u)|
        infeas_p = norm(pgrad.ravel(), ord=np.inf) \
            + norm(q0grad.ravel(), ord=np.inf) \
            + norm(q1grad.ravel(), ord=np.inf) \
            + norm(np.fmax(0.0, -ukp1.ravel()), ord=np.inf)

    return obj_p, infeas_p

def compute_dual_obj(ukp1, vkp1, wkp1, w0kp1,
                     pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1,
                     lbd, f, Y, b_sph, constraint_u, uconstrloc,
                     dataterm, avgskips, g_norms):
    ugrad = ukp1.copy()
    vgrad = vkp1.copy()
    wgrad = wkp1.copy()
    w0grad = w0kp1.copy()

    # ugrad = b q0' - q1 - diag(b) f + diag(b) D' p (quadratic)
    # ugrad = b q0' - q1 + diag(b) p0 + diag(b) D' p (W1)
    # vgrad = Y'q1
    # wgrad = Ag - BPp
    # w0grad = Ag0 - BPp0 (W1)
    manifold_op_adjoint(
        ugrad, vgrad, wgrad, w0grad,
        pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1,
        b_sph, f, Y, dataterm, avgskips
    )

    if dataterm == "quadratic":
        # obj_d = -\sum_i q0_i + 0.5*b*[f^2 - min(0, q0 + D'p - f)^2]
        l_labels = ugrad.shape[0]
        result = np.einsum('k,ki->', 0.5*b_sph.b, f.reshape(l_labels, -1)**2) \
                - np.einsum('k,ki->', 0.5/b_sph.b, np.fmin(0.0, ugrad.reshape(l_labels, -1))**2)
    elif dataterm == "W1":
        # obj_d = -\sum_i q0_i - <f,p0>_b
        result = -np.einsum('ki,ki->',
            f.reshape(f.shape[0], -1),
            np.einsum('k,ki->ki', b_sph.b, p0kp1)
        )

    obj_d = -np.sum(q0kp1)*b_sph.b_precond + result

    if dataterm == "W1":
        # infeas_d = |Y'q1| + |Ag - BPp| + |Ag0 - BPp0|
        #          + |max(0, |g| - lbd)| + |max(0, |g0| - 1.0)|
        #          + |max(0, -ugrad)|
        g0_norms = g_norms.copy()
        norms_spectral(gkp1, g_norms)
        norms_spectral(g0kp1[:,:,:,np.newaxis], g0_norms)
        infeas_d = norm(vgrad.ravel(), ord=np.inf) \
                + norm(wgrad.ravel(), ord=np.inf) \
                + norm(w0grad.ravel(), ord=np.inf) \
                + norm(np.fmax(0.0, g_norms - lbd), ord=np.inf) \
                + norm(np.fmax(0.0, g0_norms - 1.0), ord=np.inf) \
                + norm(np.fmax(0.0, -ugrad.ravel()), ord=np.inf)
    else:
        # infeas_d = |Y'q1| + |Ag - BPp| + |max(0, |g| - lbd)|
        norms_spectral(gkp1, g_norms)
        infeas_d = norm(vgrad.ravel(), ord=np.inf) \
                + norm(wgrad.ravel(), ord=np.inf) \
                + norm(np.fmax(0, g_norms - lbd), ord=np.inf)

    return obj_d, infeas_d


def manifold_op(u, v, w, w0, pgrad, ggrad, q0grad, q1grad, p0grad, g0grad,
                b_sph, f, Y, dataterm, avgskips):
    """ Apply the linear operator in the model to (u,v,w,w0).

    Args:
        u : numpy array of shape (l_labels, imagedims...)
        v : numpy array of shape (l_shm, n_image)
        w : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        w0 : numpy array of shape (n_image, m_gradients, s_manifold)
        pgrad : numpy array of shape (l_labels, d_image, n_image)
        ggrad : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        q0grad : numpy array of shape (n_image)
        q1grad : numpy array of shape (l_labels, n_image)
        p0grad : numpy array of shape (l_labels, n_image)
        g0grad : numpy array of shape (n_image, m_gradients, s_manifold)
        avgskips : output of `staggered_diff_avgskips(imagedims)`
    Returns:
        nothing, the result is written to the given arrays `pgrad`, `ggrad`,
        `q0grad`, `q1grad`, `p0grad` and `g0grad`.
    """

    l_labels = u.shape[0]
    imagedims = u.shape[1:]
    l_shm = v.shape[0]
    n_image, m_gradients, s_manifold, d_image = w.shape
    r_points = b_sph.mdims['r_points']

    pgrad[:] = 0

    # pgrad += diag(b) D u (D is the gradient on a staggered grid)
    gradient(pgrad, u, b_sph.b, avgskips)

    # pgrad_t^i += - P^j' B^j' w_t^ij
    _apply_PB(pgrad, b_sph.P, b_sph.B, w)

    # ggrad^ij = A^j' w^ij
    np.einsum('jlm,ijlt->ijmt', b_sph.A, w, out=ggrad)

    # q0grad = b'u - 1
    np.einsum('i,ij->j', b_sph.b, u.reshape(l_labels, n_image), out=q0grad)
    q0grad -= 1.0
    q0grad *= b_sph.b_precond

    # q1grad = Yv - u
    np.einsum('km,mi->ki', Y, v, out=q1grad)
    q1grad -= u.reshape(l_labels, n_image)

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

def manifold_op_adjoint(ugrad, vgrad, wgrad, w0grad, p, g, q0, q1, p0, g0,
                        b_sph, f, Y, dataterm, avgskips):
    """ Apply the adjoint linear operator in the model to (p,g,q0,q1,p0,g0).

    Args:
        ugrad : numpy array of shape (l_labels, imagedims...)
        vgrad : numpy array of shape (l_shm, n_image)
        wgrad : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        w0grad : numpy array of shape (n_image, m_gradients, s_manifold)
        p : numpy array of shape (l_labels, d_image, n_image)
        g : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        q0 : numpy array of shape (n_image)
        q1 : numpy array of shape (l_labels, n_image)
        p0 : numpy array of shape (l_labels, n_image)
        g0 : numpy array of shape (n_image, m_gradients, s_manifold)
        avgskips : output of `staggered_diff_avgskips(imagedims)`
    Returns:
        nothing, the result is written to the given arrays `ugrad`, `vgrad`,
        `wgrad` and `w0grad`.
    """

    l_labels = ugrad.shape[0]
    imagedims = ugrad.shape[1:]
    l_shm = vgrad.shape[0]
    n_image, m_gradients, s_manifold, d_image = wgrad.shape
    r_points = b_sph.mdims['r_points']

    # ugrad = b q0' - q1
    ugrad_flat = ugrad.reshape(l_labels, -1)
    np.einsum('k,i->ki', b_sph.b, b_sph.b_precond*q0, out=ugrad_flat)
    ugrad_flat -= q1

    if dataterm == "quadratic":
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

    # vgrad = Y'q1
    np.einsum('km,ki->mi', Y, q1, out=vgrad)

    # wgrad^ij = A^j g^ij
    np.einsum('jlm,ijmt->ijlt', b_sph.A, g, out=wgrad)

    # wgrad_t^ij += -B^j P^j p_t^i
    wgrad -= np.einsum('jlm,jmti->ijlt', b_sph.B, p[b_sph.P])
