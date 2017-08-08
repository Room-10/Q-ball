
from qball.tools.norm import project_gradients, norms_spectral, norms_nuclear
from qball.tools.diff import gradient, divergence

import numpy as np
from numpy.linalg import norm
from numba import jit

def compute_primal_obj(xk, xbark, xkp1, yk, ykp1,
                       sigma, tau, lbd, theta, dataterm_factor,
                       b_sph, f, Y, constraint_u, uconstrloc,
                       dataterm, avgskips, g_norms):
    ukp1, vkp1, wkp1, w0kp1 = xkp1.vars()
    ygrad = ykp1.copy()
    pgrad, ggrad, q0grad, q1grad, p0grad, g0grad = ygrad.vars()

    # pgrad = diag(b) Du - P'B'w
    # ggrad^ij = A^j' w^ij
    # q0grad = b'u - 1
    # q1grad = Yv - u
    # p0grad = diag(b) (u-f) - P'B'w0 (W1)
    # g0grad^ij = A^j' w0^ij (W1)
    manifold_op(xkp1, ygrad, b_sph, f, Y, dataterm, avgskips)
    p0grad -= np.einsum('k,ki->ki', b_sph.b, f.reshape(f.shape[0], -1))
    q0grad -= 1.0
    q0grad *= b_sph.b_precond
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

def compute_dual_obj(xk, xbark, xkp1, yk, ykp1,
                     sigma, tau, lbd, theta, dataterm_factor,
                     b_sph, f, Y, constraint_u, uconstrloc,
                     dataterm, avgskips, g_norms):
    pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1 = ykp1.vars()
    xgrad = xkp1.copy()
    ugrad, vgrad, wgrad, w0grad = xgrad.vars()

    # ugrad = b q0' - q1 - diag(b) f + diag(b) D' p (quadratic)
    # ugrad = b q0' - q1 + diag(b) p0 + diag(b) D' p (W1)
    # vgrad = Y'q1
    # wgrad = Ag - BPp
    # w0grad = Ag0 - BPp0 (W1)
    manifold_op_adjoint(xgrad, ykp1, b_sph, f, Y, dataterm, avgskips)

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

def pd_iteration_step(xk, xbark, xkp1, yk, ykp1,
                      sigma, tau, lbd, theta, dataterm_factor,
                      b_sph, f, Y, constraint_u, uconstrloc,
                      dataterm, avgskips, g_norms):
    # primals
    manifold_op_adjoint(xkp1, ykp1, b_sph, f, Y, dataterm, avgskips)
    xkp1[:] = xk - tau*xkp1
    # prox
    ukp1 = xkp1['u']
    ukp1[:] = np.einsum('k,k...->k...', dataterm_factor, ukp1)
    ukp1[:] = np.fmax(0.0, ukp1)
    ukp1[:,uconstrloc] = constraint_u[:,uconstrloc]
    xbark[:] = (1 + theta)*xkp1 - theta*xk
    xk[:] = xkp1

    # duals
    manifold_op(xbark, ykp1, b_sph, f, Y, dataterm, avgskips)
    ykp1[:] = yk + sigma*ykp1
    # prox
    pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1 = ykp1.vars()
    project_gradients(gkp1, lbd, g_norms)
    q0kp1 -= sigma*b_sph.b_precond
    p0kp1 -= sigma*np.einsum('k,ki->ki', b_sph.b, f.reshape(f.shape[0], -1))
    project_gradients(g0kp1[:,:,:,np.newaxis], 1.0, g_norms)
    yk[:] = ykp1

    """
    The Goldstein adaptive stepsizes are very memory intensive.
    They require xk, xkp1, xgradk, xgradkp1.
    That's double the memory we currently use.
    """

def manifold_op(x, ygrad, b_sph, f, Y, dataterm, avgskips):
    """ Apply the linear operator in the model to x.

    Args:
        x : primal variable
        ygrad : dual variable
        ... : more constants
    Returns:
        nothing, the result is written to the given `ygrad`.
    """
    u, v, w, w0 = x.vars()
    pgrad, ggrad, q0grad, q1grad, p0grad, g0grad = ygrad.vars()

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

    # q0grad = b'u
    np.einsum('i,ij->j', b_sph.b, u.reshape(l_labels, n_image), out=q0grad)
    q0grad *= b_sph.b_precond

    # q1grad = Yv - u
    np.einsum('km,mi->ki', Y, v, out=q1grad)
    q1grad -= u.reshape(l_labels, n_image)

    if dataterm == "W1":
        # p0grad = diag(b) u
        np.einsum('k,ki->ki', b_sph.b, u.reshape(l_labels, -1), out=p0grad)
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

def manifold_op_adjoint(xgrad, y, b_sph, f, Y, dataterm, avgskips):
    """ Apply the adjoint linear operator in the model to y

    Args:
        xgrad : primal variable
        y : dual variable
        ... : more constants
    Returns:
        nothing, the result is written to the given `xgrad`.
    """
    ugrad, vgrad, wgrad, w0grad = xgrad.vars()
    p, g, q0, q1, p0, g0 = y.vars()

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
