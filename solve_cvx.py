
import numpy as np
import cvxpy as cvx

from tools import normalize_odf
from tools_cvx import cvxVariable, sparse_div_op, cvxOp
from manifold_sphere import load_sphere

def l2_w1tv_fitting(data, gtab, sampling_matrix, model_matrix, lbd=50.0):
    b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
    b_sph = load_sphere(vecs=b_vecs)

    l_labels, l_shm = sampling_matrix.shape
    m_gradients = b_sph.mdims['m_gradients']
    imagedims = data.shape[:-1]
    d_image = len(imagedims)
    n_image = np.prod(imagedims)

    F_data = np.log(-np.log(data)).reshape(-1, l_labels).T
    F_mean = np.einsum('ki,k->i', F_data, b_sph.b)/(4*np.pi)
    F_data -= F_mean

    p  = cvxVariable(l_labels, d_image, n_image)
    g  = cvxVariable(n_image, m_gradients, d_image, 2)
    q0 = cvxVariable(n_image)
    q1 = cvxVariable(l_labels, n_image)
    q2 = cvxVariable(l_labels, n_image)

    obj = cvx.Maximize(
          0.5*cvx.sum_entries(cvx.diag(b_sph.b)*cvx.square(F_data))
        - 0.5*cvx.sum_entries(
            cvx.diag(1.0/b_sph.b)*cvx.square(q2 + cvx.diag(b_sph.b)*F_data)
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
                    Aj*g[i][j][t,:].T == sum([Bj[:,m]*p[Pj[m]][t,i] for m in range(3)])
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
            Yk = cvx.vec(sampling_matrix[:,k])
            v_constr.append(
                Yk.T*(model_matrix[k]*q2[:,i] + q1[:,i]) == 0
            )
    constraints += v_constr

    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=True)

    u1 = np.zeros((l_labels, n_image), order='C')
    for k in range(l_labels):
        for i in range(n_image):
            u1[k,i] = -u1_constr[k*n_image+i].dual_value

    v = np.zeros((l_shm, n_image), order='C')
    for k in range(l_shm):
        for i in range(n_image):
            v[k,i] = -v_constr[k*n_image+i].dual_value

    v = v.T.reshape(imagedims + (l_shm,))
    v[..., 0] = .5 / np.sqrt(np.pi) # == CsaOdfModel._n0_const
    u = u1.T.reshape(imagedims + (l_labels,))
    return u, v

def w1_tv_regularization(f, gtab, sampling_matrix=None, lbd=10.0):
    b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
    b_sph = load_sphere(vecs=b_vecs)

    l_labels = b_sph.mdims['l_labels']
    m_gradients = b_sph.mdims['m_gradients']
    assert(f.shape[0] == l_labels)
    imagedims = f.shape[1:]
    d_image = len(imagedims)
    n_image = np.prod(imagedims)

    p  = cvxVariable(l_labels, d_image, n_image)
    g  = cvxVariable(n_image, m_gradients, d_image, 2)
    q0 = cvxVariable(n_image)
    q1 = cvxVariable(l_labels, n_image)
    p0 = cvxVariable(l_labels, n_image)
    g0 = cvxVariable(n_image, m_gradients, 2)

    Y = np.eye(l_labels)
    if sampling_matrix is not None:
        Y = sampling_matrix
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
                    Aj*g[i][j][t,:].T == sum([Bj[:,m]*p[Pj[m]][t,i] for m in range(3)])
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
    prob.solve(verbose=True)

    v = np.zeros((l_shm, n_image), order='C')
    for k in range(l_shm):
        for i in range(n_image):
            v[k,i] = -v_constr[k*n_image+i].dual_value
    v = v.T.reshape(imagedims + (l_shm,))

    u = np.zeros((l_labels, n_image), order='C')
    for k in range(l_labels):
        for i in range(n_image):
            u[k,i] = u_constr[k*n_image+i].dual_value
    u = u.T.reshape(imagedims + (l_labels,))

    return u, v
