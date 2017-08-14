
from qball.tools import normalize_odf
from qball.tools.blocks import BlockVar
from qball.tools.cvx import cvxVariable, sparse_div_op, cvxOp
from qball.sphere import load_sphere

import numpy as np
import cvxpy as cvx

import logging

def qball_regularization(f, gtab, sampling_matrix, lbd=10.0):
    b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
    b_sph = load_sphere(vecs=b_vecs)

    imagedims = f.shape[:-1]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    s_manifold = 2
    l_labels = b_sph.mdims['l_labels']
    m_gradients = b_sph.mdims['m_gradients']
    assert(f.shape[-1] == l_labels)

    f_flat = f.reshape(-1, l_labels).T
    f = np.array(f_flat.reshape((l_labels,) + imagedims), order='C')
    normalize_odf(f, b_sph.b)
    f_flat = f.reshape(l_labels, n_image)

    Y = np.zeros(sampling_matrix.shape, order='C')
    Y[:] = sampling_matrix
    l_shm = Y.shape[1]

    logging.info("Solving ({l_labels} labels, m={m}; img: {imagedims}; " \
                 "lambda={lbd:.3g}) using CVX...".format(
        lbd=lbd,
        m=m_gradients,
        l_labels=l_labels,
        imagedims="x".join(map(str,imagedims)),
    ))

    p  = cvxVariable(l_labels, d_image, n_image)
    g  = cvxVariable(n_image, m_gradients, 2, d_image)
    q0 = cvxVariable(n_image)
    q1 = cvxVariable(l_labels, n_image)
    p0 = cvxVariable(l_labels, n_image)
    g0 = cvxVariable(n_image, m_gradients, 2)

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

    # Store result in block variables
    x = BlockVar(
        ('u', (l_labels,) + imagedims),
        ('v', (l_shm, n_image)),
        ('w', (n_image, m_gradients, s_manifold, d_image)),
        ('w0', (n_image, m_gradients, s_manifold))
    )

    y = BlockVar(
        ('p', (l_labels, d_image, n_image)),
        ('g', (n_image, m_gradients, s_manifold, d_image)),
        ('q0', (n_image,)),
        ('q1', (l_labels, n_image)),
        ('p0', (l_labels, n_image)),
        ('g0', (n_image, m_gradients, s_manifold))
    )

    for k in range(l_labels):
        y['p'][k,:] = p[k].value

    for i in range(n_image):
        for j in range(m_gradients):
            y['g'][i,j,:,:] = g[i][j].value

    y['q0'][:] = q0.value.ravel()
    y['q1'][:,:] = q1.value
    y['p0'][:,:] = p0.value

    for i in range(n_image):
        y['g0'][i,:,:] = g0[i].value

    for k in range(l_shm):
        for i in range(n_image):
            x['v'][k,i] = -v_constr[k*n_image+i].dual_value

    u_flat = x['u'].reshape((l_labels, n_image))
    for k in range(l_labels):
        for i in range(n_image):
            u_flat[k,i] = u_constr[k*n_image+i].dual_value

    for j in range(m_gradients):
        for i in range(n_image):
            for t in range(d_image):
                x['w'][i,j,:,t] = w_constr[(j*n_image + i)*d_image + t].dual_value.ravel()

    for j in range(m_gradients):
        for i in range(n_image):
            x['w0'][i,j,:] = w0_constr[j*n_image + i].dual_value.ravel()

    logging.debug("{}: objd = {: 9.6g}".format(prob.status, prob.value))
    return (x,y), { 'objp': prob.value, 'status': prob.status }
