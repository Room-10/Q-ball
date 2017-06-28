
from tools_norm import project_gradients, norms_spectral, norms_nuclear
from tools_diff import gradient, divergence

import numpy as np
from numpy.linalg import norm
from numba import jit

def compute_primal_obj(u1k, u2k, vk, wk,
                       u1bark, u2bark, vbark, wbark,
                       u1kp1, u2kp1, vkp1, wkp1,
                       pk, gk, q0k, q1k, q2k,
                       pkp1, gkp1, q0kp1, q1kp1, q2kp1,
                       sigma, tau, lbd, theta, dataterm_factor,
                       b_sph, f, Y, M, constraint_u, uconstrloc,
                       avgskips, g_norms):
    pgrad = pkp1.copy()
    ggrad = gkp1.copy()
    q0grad = q0kp1.copy()
    q1grad = q1kp1.copy()
    q2grad = q2kp1.copy()

    # pgrad = diag(b) Du1 - P'B'w
    # ggrad^ij = A^j' w^ij
    # q0grad = b'u1 - 1
    # q1grad = Yv - u1
    # q2grad = YMv - u2
    manifold_op(
        u1kp1, u2kp1, vkp1, wkp1,
        pgrad, ggrad, q0grad, q1grad, q2grad,
        b_sph, f, Y, M, avgskips
    )
    norms_nuclear(ggrad, g_norms)

    # obj_p = 0.5*<u2-f,u2-f>_b + lbd*\sum_ij |A^j' w^ij|
    obj_p = np.einsum('k,ki->', b_sph.b, 0.5*(u2kp1 - f)**2) + lbd * g_norms.sum()

    # infeas_p = |diag(b) Du1 - P'B'w| + |b'u1 - 1|
    #          + |Yv - u1| + |YMv - u2| + |max(0,-u1)|
    infeas_p = norm(pgrad.ravel(), ord=np.inf) \
        + norm(q0grad.ravel(), ord=np.inf) \
        + norm(q1grad.ravel(), ord=np.inf) \
        + norm(q2grad.ravel(), ord=np.inf) \
        + norm(np.fmax(0.0, -u1kp1.ravel()), ord=np.inf)

    return obj_p, infeas_p

def compute_dual_obj(u1k, u2k, vk, wk,
                     u1bark, u2bark, vbark, wbark,
                     u1kp1, u2kp1, vkp1, wkp1,
                     pk, gk, q0k, q1k, q2k,
                     pkp1, gkp1, q0kp1, q1kp1, q2kp1,
                     sigma, tau, lbd, theta, dataterm_factor,
                     b_sph, f, Y, M, constraint_u, uconstrloc,
                     avgskips, g_norms):
    u1grad = u1kp1.copy()
    u2grad = u2kp1.copy()
    vgrad = vkp1.copy()
    wgrad = wkp1.copy()

    # u1grad = b q0' - q1 + diag(b) D' p
    # u2grad = -diag(b) f - q2
    # vgrad = Y'q1 + M Y'q2
    # wgrad = Ag - BPp
    manifold_op_adjoint(
        u1grad, u2grad, vgrad, wgrad,
        pkp1, gkp1, q0kp1, q1kp1, q2kp1,
        b_sph, f, Y, M, avgskips
    )

    # obj_d = -\sum_i q0_i + 0.5*b*[f^2 - (diag(1/b) q2 + f)^2]
    obj_d = -np.sum(q0kp1)*b_sph.b_precond \
          + np.einsum('k,ki->', 0.5*b_sph.b, f**2) \
          - np.einsum('k,ki->', 0.5/b_sph.b, u2grad**2)

    # infeas_d = |Y'q1 + M Y'q2| + |Ag - BPp| + |max(0, |g| - lbd)|
    norms_spectral(gkp1, g_norms)
    infeas_d = norm(vgrad.ravel(), ord=np.inf) \
            + norm(wgrad.ravel(), ord=np.inf) \
            + norm(np.fmax(0, g_norms - lbd), ord=np.inf) \
            + norm(np.fmax(0.0, -u1grad.ravel()), ord=np.inf)

    return obj_d, infeas_d

def pd_iteration_step(u1k, u2k, vk, wk,
                      u1bark, u2bark, vbark, wbark,
                      u1kp1, u2kp1, vkp1, wkp1,
                      pk, gk, q0k, q1k, q2k,
                      pkp1, gkp1, q0kp1, q1kp1, q2kp1,
                      sigma, tau, lbd, theta, dataterm_factor,
                      b_sph, f, Y, M, constraint_u, uconstrloc,
                      avgskips, g_norms):
    # duals
    manifold_op(
        u1bark, u2bark, vbark, wbark,
        pkp1, gkp1, q0kp1, q1kp1, q2kp1,
        b_sph, f, Y, M, avgskips
    )
    pkp1[:] = pk + sigma*pkp1
    gkp1[:] = gk + sigma*gkp1
    q0kp1[:] = q0k + sigma*q0kp1
    q1kp1[:] = q1k + sigma*q1kp1
    q2kp1[:] = q2k + sigma*q2kp1
    project_gradients(gkp1, lbd, g_norms)
    # update
    pk[:] = pkp1
    gk[:] = gkp1
    q0k[:] = q0kp1
    q1k[:] = q1kp1
    q2k[:] = q2kp1

    # primals
    manifold_op_adjoint(
        u1kp1, u2kp1, vkp1, wkp1,
        pkp1, gkp1, q0kp1, q1kp1, q2kp1,
        b_sph, f, Y, M, avgskips
    )
    u1kp1[:] = u1k - tau*u1kp1
    u1kp1[:] = np.fmax(0.0, u1kp1)
    u1kp1[:,uconstrloc] = constraint_u[:,uconstrloc]
    np.einsum('k,k...->k...', dataterm_factor, u2k - tau*u2kp1, out=u2kp1)
    vkp1[:] = vk - tau*vkp1
    wkp1[:] = wk - tau*wkp1
    # overrelaxation
    u1bark[:] = u1kp1 + theta * (u1kp1 - u1k)
    u2bark[:] = u2kp1 + theta * (u2kp1 - u2k)
    vbark[:] = vkp1 + theta * (vkp1 - vk)
    wbark[:] = wkp1 + theta * (wkp1 - wk)
    # update
    u1k[:] = u1kp1
    u2k[:] = u2kp1
    vk[:] = vkp1
    wk[:] = wkp1


def manifold_op(u1, u2, v, w, pgrad, ggrad, q0grad, q1grad, q2grad,
                b_sph, f, Y, M, avgskips):
    """ Apply the linear operator in the model to (u1,u2,v,w).

    Args:
        u1 : numpy array of shape (l_labels, imagedims...)
        u2 : numpy array of shape (l_labels, n_image)
        v : numpy array of shape (l_shm, n_image)
        w : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        pgrad : numpy array of shape (l_labels, d_image, n_image)
        ggrad : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        q0grad : numpy array of shape (n_image)
        q1grad : numpy array of shape (l_labels, n_image)
        q2grad : numpy array of shape (l_labels, n_image)
        avgskips : output of `staggered_diff_avgskips(imagedims)`
    Returns:
        nothing, the result is written to the given arrays `pgrad`, `ggrad`,
        `q0grad`, `q1grad`, and `q2grad`.
    """

    l_labels = u1.shape[0]
    imagedims = u1.shape[1:]
    l_shm = v.shape[0]
    n_image, m_gradients, s_manifold, d_image = w.shape
    r_points = b_sph.mdims['r_points']

    pgrad[:] = 0

    # pgrad += diag(b) D u (D is the gradient on a staggered grid)
    gradient(pgrad, u1, b_sph.b, avgskips)

    # pgrad_t^i += - P^j' B^j' w_t^ij
    _apply_PB(pgrad, b_sph.P, b_sph.B, w)

    # ggrad^ij = A^j' w^ij
    np.einsum('jlm,ijlt->ijmt', b_sph.A, w, out=ggrad)

    # q0grad = b'u - 1
    np.einsum('i,ij->j', b_sph.b, u1.reshape(l_labels, n_image), out=q0grad)
    q0grad -= 1.0
    q0grad *= b_sph.b_precond

    # q1grad = Yv - u1
    np.einsum('km,mi->ki', Y, v, out=q1grad)
    q1grad -= u1.reshape(l_labels, n_image)

    # q2grad = YMv - u2
    np.einsum('km,mi->ki', Y, np.einsum('m,mi->mi', M, v), out=q2grad)
    q2grad -= u2

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

def manifold_op_adjoint(u1grad, u2grad, vgrad, wgrad, p, g, q0, q1, q2,
                        b_sph, f, Y, M, avgskips):
    """ Apply the adjoint linear operator in the model to (p,g,q0,q1,q2).

    Args:
        u1grad : numpy array of shape (l_labels, imagedims...)
        u2grad : numpy array of shape (l_labels, n_image)
        vgrad : numpy array of shape (l_shm, n_image)
        wgrad : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        p : numpy array of shape (l_labels, d_image, n_image)
        g : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
        q0 : numpy array of shape (n_image)
        q1 : numpy array of shape (l_labels, n_image)
        q2 : numpy array of shape (l_labels, n_image)
        avgskips : output of `staggered_diff_avgskips(imagedims)`
    Returns:
        nothing, the result is written to the given arrays `u1grad`, `u2grad`,
        `vgrad` and `wgrad`.
    """

    l_labels = u1grad.shape[0]
    imagedims = u1grad.shape[1:]
    l_shm = vgrad.shape[0]
    n_image, m_gradients, s_manifold, d_image = wgrad.shape
    r_points = b_sph.mdims['r_points']

    # u1grad = b q0' - q1
    u1grad_flat = u1grad.reshape(l_labels, -1)
    np.einsum('k,i->ki', b_sph.b, b_sph.b_precond*q0, out=u1grad_flat)
    u1grad_flat -= q1

    # u1grad += diag(b) D' p (where D' = -div with Dirichlet boundary)
    divergence(p, u1grad, b_sph.b, avgskips)

    # u2grad = -q2 - diag(b) f
    u2grad[:] = -q2
    u2grad -= np.einsum('k,ki->ki', b_sph.b, f)

    # vgrad = Y'q1 + M Y'q2
    np.einsum('km,ki->mi', Y, q1, out=vgrad)
    vgrad += np.einsum('m,mi->mi', M, np.einsum('km,ki->mi', Y, q2))

    # wgrad^ij = A^j g^ij
    np.einsum('jlm,ijmt->ijlt', b_sph.A, g, out=wgrad)

    # wgrad_t^ij += -B^j P^j p_t^i
    wgrad -= np.einsum('jlm,jmti->ijlt', b_sph.B, p[b_sph.P])
