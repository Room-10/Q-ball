
import numpy as np
import cvxpy as cvx

def w1_dist(f, u, mf, verbose=False):
    """ Determine W^1(f_i,u_i) of finite measures on a manifold mf

    Args:
        f, u : arrays of column-wise finite measures on mf
        mf : manifold
    Returns:
        the Wasserstein-1-distances W^1(f_i,u_i)
    """

    n_image = f.shape[1]
    l_labels = mf.mdims['l_labels']
    s_manifold = mf.mdims['s_manifold']
    m_gradients = mf.mdims['m_gradients']
    assert l_labels == f.shape[0]

    fmu = np.einsum('k,ki->ki', mf.b, f - u)

    results = np.zeros(n_image)
    for i in range(n_image):
        p = cvx.Variable(l_labels)
        g = cvx.Variable(m_gradients, s_manifold)
        obj = cvx.Maximize(fmu[:,i]*p)

        constraints = []
        for j in range(m_gradients):
            constraints.append(mf.A[j,:,:]*g[j,:].T == mf.B[j,:,:]*p[mf.P[j,:]])
            constraints.append(cvx.norm(g[j,:].T, 2) <= 1)

        prob = cvx.Problem(obj, constraints)
        prob.solve()
        results[i] = obj.value

    return results
