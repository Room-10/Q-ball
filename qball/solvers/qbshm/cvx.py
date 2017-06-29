
from tools import normalize_odf
from qball.tools.cvx import cvxVariable, sparse_div_op, cvxOp
from qball.sphere import load_sphere

import numpy as np
import cvxpy as cvx

import logging

def w1_tv_regularization(f, gtab, sampling_matrix, lbd=10.0):
    b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
    b_sph = load_sphere(vecs=b_vecs)

    l_labels = b_sph.mdims['l_labels']
    m_gradients = b_sph.mdims['m_gradients']
    assert(f.shape[0] == l_labels)
    imagedims = f.shape[1:]
    d_image = len(imagedims)
    n_image = np.prod(imagedims)

    p  = cvxVariable(l_labels, d_image, n_image)
    g  = cvxVariable(n_image, m_gradients, 2, d_image)
    q0 = cvxVariable(n_image)
    q1 = cvxVariable(l_labels, n_image)
    p0 = cvxVariable(l_labels, n_image)
    g0 = cvxVariable(n_image, m_gradients, 2)

    Y = np.zeros(sampling_matrix.shape, order='C')
    Y[:] = sampling_matrix
    l_shm = Y.shape[1]

    normalize_odf(f, b_sph.b)
    f_flat = f.reshape(l_labels, n_image)

    obj = cvx.Maximize(
        - cvx.vec(f_flat).T*cvx.vec(cvx.diag(b_sph.b)*p0)
        - cvx.sum_entries(q0)
    )

    div_op = sparse_div_op(imagedims)

    constraints = []
    for i in range(n_image):
        for j in range(m_gradients):
            constraints.append(cvx.norm(g[i][j], 2) <= lbd)
            constraints.append(cvx.norm(g0[i][j,:], 2) <= 1.0)

    w0_constr = []
    w_constr = []
    for j in range(m_gradients):
        Aj = b_sph.A[j,:,:]
        Bj = b_sph.B[j,:,:]
        Pj = b_sph.P[j,:]
        for i in range(n_image):
            w0_constr.append(Aj*g0[i][j,:].T == Bj*p0[Pj,i])
            for t in range(d_image):
                w_constr.append(
                    Aj*g[i][j][:,t] == sum([Bj[:,m]*p[Pj[m]][t,i] for m in range(3)])
                )
    constraints += w0_constr
    constraints += w_constr

    u_constr = []
    for k in range(l_labels):
        for i in range(n_image):
            u_constr.append(
                b_sph.b[k]*(q0[i] + p0[k,i] - cvxOp(div_op, p[k], i)) - q1[k,i] >= 0
            )
    constraints += u_constr

    v_constr = []
    for k in range(l_shm):
        for i in range(n_image):
            Yk = cvx.vec(Y[:,k])
            v_constr.append(
                Yk.T*q1[:,i] == 0
            )
    constraints += v_constr

    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=False)

    pk = np.zeros((l_labels, d_image, n_image), order='C')
    for k in range(l_labels):
        pk[k,:] = p[k].value
    gk = np.zeros((n_image, m_gradients, 2, d_image), order='C')
    for i in range(n_image):
        for j in range(m_gradients):
            gk[i,j,:,:] = g[i][j].value
    q0k = np.zeros(n_image)
    q0k[:] = q0.value.ravel()
    q1k = np.zeros((l_labels, n_image), order='C')
    q1k[:,:] = q1.value
    p0k = np.zeros((l_labels, n_image), order='C')
    p0k[:,:] = p0.value
    g0k = np.zeros((n_image, m_gradients, 2), order='C')
    for i in range(n_image):
        g0k[i,:,:] = g0[i].value

    vk = np.zeros((l_shm, n_image), order='C')
    for k in range(l_shm):
        for i in range(n_image):
            vk[k,i] = -v_constr[k*n_image+i].dual_value

    uk = np.zeros((l_labels, n_image), order='C')
    for k in range(l_labels):
        for i in range(n_image):
            uk[k,i] = u_constr[k*n_image+i].dual_value
    uk = uk.reshape((l_labels,) + imagedims)

    wk = np.zeros((n_image, m_gradients, 2, d_image), order='C')
    for j in range(m_gradients):
        for i in range(n_image):
            for t in range(d_image):
                wk[i,j,:,t] = w_constr[(j*n_image + i)*d_image + t].dual_value.ravel()

    w0k = np.zeros((n_image, m_gradients, 2), order='C')
    for j in range(m_gradients):
        for i in range(n_image):
            w0k[i,j,:] = w0_constr[j*n_image + i].dual_value.ravel()

    logging.debug("{}: objd = {: 9.6g}".format(prob.status, prob.value))
    return (uk, vk, wk, w0k, pk, gk, q0k, q1k, p0k, g0k), {
        'objp': prob.value,
        'status': prob.status
    }
