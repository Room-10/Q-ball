
from tools_cvx import cvxVariable, sparse_div_op, cvxOp
from manifold_sphere import load_sphere

import numpy as np
import cvxpy as cvx

import logging

def l2_w1tv_fitting(data, gtab, sampling_matrix, model_matrix, lbd=50.0):
    b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
    b_sph = load_sphere(vecs=b_vecs)

    l_labels = b_sph.mdims['l_labels']
    m_gradients = b_sph.mdims['m_gradients']
    assert(data.shape[-1] == l_labels)
    imagedims = data.shape[:-1]
    d_image = len(imagedims)
    n_image = np.prod(imagedims)

    f = np.zeros((l_labels, n_image), order='C')
    f[:] = np.log(-np.log(data)).reshape(-1, l_labels).T
    f_mean = np.einsum('ki,k->i', f, b_sph.b)/(4*np.pi)
    f -= f_mean

    p  = cvxVariable(l_labels, d_image, n_image)
    g  = cvxVariable(n_image, m_gradients, 2, d_image)
    q0 = cvxVariable(n_image)
    q1 = cvxVariable(l_labels, n_image)
    q2 = cvxVariable(l_labels, n_image)

    Y = np.zeros(sampling_matrix.shape, order='C')
    Y[:] = sampling_matrix
    l_shm = Y.shape[1]
    M = model_matrix
    assert(M.size == l_shm)

    obj = cvx.Maximize(
          0.5*cvx.sum_entries(cvx.diag(b_sph.b)*cvx.square(f))
        - 0.5*cvx.sum_entries(
            cvx.diag(1.0/b_sph.b)*cvx.square(q2 + cvx.diag(b_sph.b)*f)
        ) - cvx.sum_entries(q0)
    )

    div_op = sparse_div_op(imagedims)

    constraints = []
    for i in range(n_image):
        for j in range(m_gradients):
            constraints.append(cvx.norm(g[i][j], 2) <= lbd)

    w_constr = []
    for j in range(m_gradients):
        Aj = b_sph.A[j,:,:]
        Bj = b_sph.B[j,:,:]
        Pj = b_sph.P[j,:]
        for i in range(n_image):
            for t in range(d_image):
                w_constr.append(
                    Aj*g[i][j][:,t] == sum([Bj[:,m]*p[Pj[m]][t,i] for m in range(3)])
                )
    constraints += w_constr

    u1_constr = []
    for k in range(l_labels):
        for i in range(n_image):
            u1_constr.append(
               b_sph.b[k]*(q0[i] - cvxOp(div_op, p[k], i)) - q1[k,i] >= 0
            )
    constraints += u1_constr

    v_constr = []
    for k in range(l_shm):
        for i in range(n_image):
            Yk = cvx.vec(Y[:,k])
            v_constr.append(
                Yk.T*(M[k]*q2[:,i] + q1[:,i]) == 0
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
    q2k = np.zeros((l_labels, n_image), order='C')
    q2k[:,:] = q2.value

    vk = np.zeros((l_shm, n_image), order='C')
    for k in range(l_shm):
        for i in range(n_image):
            vk[k,i] = -v_constr[k*n_image+i].dual_value

    u1k = np.zeros((l_labels, n_image), order='C')
    for k in range(l_labels):
        for i in range(n_image):
            u1k[k,i] = -u1_constr[k*n_image+i].dual_value
    u1k = u1k.reshape((l_labels,) + imagedims)

    u2k = np.zeros((l_labels, n_image), order='C')
    np.einsum('km,mi->ki', sampling_matrix,
        np.einsum('m,mi->mi', model_matrix, vk), out=u2k)

    wk = np.zeros((n_image, m_gradients, 2, d_image), order='C')
    for j in range(m_gradients):
        for i in range(n_image):
            for t in range(d_image):
                wk[i,j,:,t] = w_constr[(j*n_image + i)*d_image + t].dual_value.ravel()

    logging.debug("{}: objd = {: 9.6g}".format(prob.status, prob.value))
    return (u1k, u2k, vk, wk, pk, gk, q0k, q1k, q2k), {
        'objp': prob.value,
        'status': prob.status
    }
