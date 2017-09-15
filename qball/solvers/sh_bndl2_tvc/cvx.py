
from qball.tools.bounds import compute_bounds
from qball.tools.blocks import BlockVar
from qball.tools.cvx import cvxVariable, sparse_div_op, cvxOp
from qball.sphere import load_sphere

import numpy as np
import cvxpy as cvx

import logging

def fit_hardi_qball(data, gtab, sampling_matrix, model_matrix, lbd=1.0):
    b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
    b_sph = load_sphere(vecs=b_vecs)

    imagedims = data.shape[:-1]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    l_labels = b_sph.mdims['l_labels']
    assert(data.shape[-1] == l_labels)

    fl, fu = compute_bounds(b_sph, data)

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

    fid_fun_dual = 0
    for k in range(l_labels):
        for i in range(n_image):
            fid_fun_dual += -1.0/b_sph.b[k]*(cvx.power(q2[k,i],2)/2 \
                         + cvx.max_elemwise(q2[k,i]*b_sph.b[k]*fl[k,i],
                             q2[k,i]*b_sph.b[k]*fu[k,i]))

    obj = cvx.Maximize(fid_fun_dual - cvx.sum_entries(q0))

    div_op = sparse_div_op(imagedims)

    constraints = []
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
    )

    for k in range(l_shm):
        y['p'][k,:] = p[k].value

    y['q0'][:] = q0.value.ravel()
    y['q1'][:,:] = q1.value
    y['q2'][:,:] = q2.value

    for k in range(l_shm):
        for i in range(n_image):
            x['v'][k,i] = v_constr[k*n_image+i].dual_value

    u1_flat = x['u1'].reshape((l_labels, -1))
    for k in range(l_labels):
        for i in range(n_image):
            u1_flat[k,i] = u1_constr[k*n_image+i].dual_value

    np.einsum('km,mi->ki', sampling_matrix,
        np.einsum('m,mi->mi', model_matrix, x['v']), out=x['u2'])

    logging.debug("{}: objd = {: 9.6g}".format(prob.status, prob.value))
    return (x,y), { 'objp': prob.value, 'status': prob.status }
