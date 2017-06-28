
from tools_cvx import cvxVariable, sparse_div_op, cvxOp
from manifold_sphere import load_sphere

import numpy as np
import cvxpy as cvx

import logging

def l2_tv_fitting(data, gtab, sampling_matrix, model_matrix, lbd=50.0):
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

    Y = np.zeros(sampling_matrix.shape, order='C')
    Y[:] = sampling_matrix
    l_shm = Y.shape[1]
    M = model_matrix
    assert(M.size == l_shm)

    p  = cvxVariable(l_labels, d_image, n_image)
    q0 = cvxVariable(n_image)
    q1 = cvxVariable(l_labels, n_image)
    q2 = cvxVariable(l_labels, n_image)

    obj = cvx.Maximize(
          0.5*cvx.sum_entries(cvx.diag(b_sph.b)*cvx.square(f))
        - 0.5*cvx.sum_entries(
            cvx.diag(1.0/b_sph.b)*cvx.square(q2 + cvx.diag(b_sph.b)*f)
        ) - cvx.sum_entries(q0)
    )

    div_op = sparse_div_op(imagedims)

    constraints = []
    for i in range(n_image):
        constraints.append(sum(cvx.sum_squares(p[k][:,i]) for k in range(l_shm)) <= lbd**2)

    u1_constr = []
    for k in range(l_labels):
        for i in range(n_image):
            u1_constr.append(
               b_sph.b[k]*q0[i] - q1[k,i] - cvxOp(div_op, p[k], i) >= 0
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

    logging.debug("{}: objd = {: 9.6g}".format(prob.status, prob.value))
    return (u1k, u2k, vk, pk, q0k, q1k, q2k), {
        'objp': prob.value,
        'status': prob.status
    }
