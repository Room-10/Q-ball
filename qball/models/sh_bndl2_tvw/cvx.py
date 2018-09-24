
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
    data_ext = data
    data = data_ext['raw'][data_ext['slice']]
    b_sph = data['b_sph']

    imagedims = data.shape[:-1]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    s_manifold = 2
    l_labels = b_sph.mdims['l_labels']
    m_gradients = b_sph.mdims['m_gradients']
    assert(data.shape[-1] == l_labels)

    alpha = model_params.get('conf_lvl', 0.9)
    compute_hardi_bounds(data_ext, alpha=alpha)
    _, fl, fu = data_ext['bounds']

    Y = np.zeros(sampling_matrix.shape, order='C')
    Y[:] = sampling_matrix
    l_shm = Y.shape[1]

    M = model_matrix
    assert(M.size == l_shm)

    logging.info("Solving ({l_labels} labels (m={m_gradients}), {l_shm} shm; " \
        "img: {imagedims}; lambda={lbd:.3g}) using CVX...".format(
        lbd=lbd,
        l_labels=l_labels,
        m_gradients=m_gradients,
        l_shm=l_shm,
        imagedims="x".join(map(str,imagedims)),
    ))

    p  = cvxVariable(l_labels, d_image, n_image)
    g  = cvxVariable(n_image, m_gradients, s_manifold, d_image)
    q0 = cvxVariable(n_image)
    q1 = cvxVariable(l_labels, n_image)
    q2 = cvxVariable(l_labels, n_image)

    fid_fun_dual = 0
    for k in range(l_labels):
        for i in range(n_image):
            fid_fun_dual += -cvx.power(q2[k,i],2)/2 \
                         - cvx.max_elemwise(q2[k,i]*fl[k,i],q2[k,i]*fu[k,i])

    obj = cvx.Maximize(fid_fun_dual - cvx.sum_entries(q0))

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

    # Store result in block variables
    x = BlockVar(
        ('u1', (l_labels,) + imagedims),
        ('u2', (l_labels, n_image)),
        ('v', (l_shm, n_image)),
        ('w', (n_image, m_gradients, s_manifold, d_image)),
    )

    y = BlockVar(
        ('p', (l_labels, d_image, n_image)),
        ('g', (n_image, m_gradients, s_manifold, d_image)),
        ('q0', (n_image,)),
        ('q1', (l_labels, n_image)),
        ('q2', (l_labels, n_image)),
    )

    for k in range(l_labels):
        y['p'][k,:] = p[k].value

    for i in range(n_image):
        for j in range(m_gradients):
            y['g'][i,j,:,:] = g[i][j].value

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

    for j in range(m_gradients):
        for i in range(n_image):
            for t in range(d_image):
                x['w'][i,j,:,t] = w_constr[(j*n_image + i)*d_image + t].dual_value.ravel()

    logging.info("{}: objd = {: 9.6g}".format(prob.status, prob.value))
    return (x,y), { 'objp': prob.value, 'status': prob.status }
