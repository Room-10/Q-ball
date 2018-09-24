
from qball.tools.bounds import compute_hardi_bounds
from qball.tools.cvx import cvxVariable, sparse_div_op, cvxOp

import numpy as np
import cvxpy as cvx

import logging

from opymize import BlockVar

def fit_hardi_qball(data, model_params, solver_params={}):
    sampling_matrix = model_params['sampling_matrix']
    model_matrix = model_params['model_matrix']
    lbd = model_params.get('lbd', 1.0)
    data = data['raw'][data['slice']]
    b_sph = data['b_sph']

    imagedims = data.shape[:-1]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    l_labels = b_sph.mdims['l_labels']
    assert(data.shape[-1] == l_labels)

    alpha = model_params.get('conf_lvl', 0.9)
    compute_hardi_bounds(data_ext, alpha=alpha)
    _, fl, fu = data_ext['bounds']

    Y = np.zeros(sampling_matrix.shape, order='C')
    Y[:] = sampling_matrix
    l_shm = Y.shape[1]

    M = model_matrix
    assert(M.size == l_shm)

    logging.info("Solving ({l_labels} labels, {l_shm} shm; " \
        "img: {imagedims}; lambda={lbd:.3g}) using CVX...".format(
        lbd=lbd,
        l_labels=l_labels,
        l_shm=l_shm,
        imagedims="x".join(map(str,imagedims)),
    ))

    p  = cvxVariable(l_shm, d_image, n_image)
    q0 = cvxVariable(n_image)
    q1 = cvxVariable(l_labels, n_image)
    q2 = cvxVariable(l_labels, n_image)
    q3 = cvxVariable(l_labels, n_image)
    q4 = cvxVariable(l_labels, n_image)

    obj = cvx.Maximize(
        cvx.vec(q3).T*cvx.vec(cvx.diag(b_sph.b)*fl)
        - cvx.vec(q4).T*cvx.vec(cvx.diag(b_sph.b)*fu)
        - cvx.sum_entries(q0)
    )

    div_op = sparse_div_op(imagedims)

    constraints = []
    for i in range(n_image):
        for k in range(l_labels):
            constraints.append(0 <= q3[k,i])
            constraints.append(q3[k,i] <= 1)
            constraints.append(0 <= q4[k,i])
            constraints.append(q4[k,i] <= 1)
            constraints.append(q4[k,i] - q3[k,i] - q2[k,i] == 0)
    for i in range(n_image):
        constraints.append(sum(cvx.sum_squares(p[k][:,i]) for k in range(l_shm)) <= lbd**2)

    u1_constr = []
    for k in range(l_labels):
        for i in range(n_image):
            u1_constr.append(
               b_sph.b[k]*q0[i] - q1[k,i] >= 0
            )
    constraints += u1_constr

    v_constr = []
    for k in range(l_shm):
        for i in range(n_image):
            Yk = cvx.vec(Y[:,k])
            v_constr.append(
                Yk.T*(M[k]*q2[:,i] + q1[:,i]) - cvxOp(div_op, p[k], i) == 0
            )
    constraints += v_constr

    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=False)

    # Store result in block variables
    x = BlockVar(
        ('u1', (l_labels,) + imagedims),
        ('u2', (l_labels, n_image)),
        ('v', (l_shm, n_image)),
    )

    y = BlockVar(
        ('p', (l_shm, d_image, n_image)),
        ('q0', (n_image,)),
        ('q1', (l_labels, n_image)),
        ('q2', (l_labels, n_image)),
        ('q3', (l_labels, n_image)),
        ('q4', (l_labels, n_image)),
    )

    for k in range(l_shm):
        y['p'][k,:] = p[k].value

    y['q0'][:] = q0.value.ravel()
    y['q1'][:,:] = q1.value
    y['q2'][:,:] = q2.value
    y['q3'][:,:] = q3.value
    y['q4'][:,:] = q4.value

    for k in range(l_shm):
        for i in range(n_image):
            x['v'][k,i] = v_constr[k*n_image+i].dual_value

    u1_flat = x['u1'].reshape((l_labels, -1))
    for k in range(l_labels):
        for i in range(n_image):
            u1_flat[k,i] = u1_constr[k*n_image+i].dual_value

    np.einsum('km,mi->ki', sampling_matrix,
        np.einsum('m,mi->mi', model_matrix, x['v']), out=x['u2'])

    logging.info("{}: objd = {: 9.6g}".format(prob.status, prob.value))
    return (x,y), { 'objp': prob.value, 'status': prob.status }
