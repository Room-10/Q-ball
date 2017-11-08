
import numpy as np
import cvxpy as cvx

from functools import partial
import multiprocessing
import os, sys

def w1_dist(f, u, mf):
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

    worker_count = multiprocessing.cpu_count()
    chunks = np.array_split(fmu, worker_count, axis=1)

    fd = sys.stderr.fileno()
    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        # ignore CUDA warnings about failed clean-up operations
        # these seem to be caused by multiprocessing's "fork"
        # (manual clean-up doesn't seem to be possible with PyCUDA)
        with open(os.devnull, 'w') as file:
            # redirect stderr to /dev/null
            sys.stderr.close()
            os.dup2(file.fileno(), fd)
            sys.stderr = os.fdopen(fd, 'w')
        try:
            # parallelize the computations on the available CPU cores
            p = multiprocessing.Pool(processes=worker_count)
            worker_partial = partial(w1_dist_worker, mf=mf)
            w_results = p.map(worker_partial, chunks)
            p.terminate()
        finally:
            # restore stderr
            sys.stderr.close()
            os.dup2(old_stderr.fileno(), fd)
            sys.stderr = os.fdopen(fd, 'w')

    results[:] = np.concatenate(w_results)

    return results

def w1_dist_worker(fmu, mf):
    n_image = fmu.shape[1]
    l_labels = mf.mdims['l_labels']
    s_manifold = mf.mdims['s_manifold']
    m_gradients = mf.mdims['m_gradients']
    assert l_labels == fmu.shape[0]

    p = cvx.Variable(l_labels)
    g = cvx.Variable(m_gradients, s_manifold)
    fmu_i = cvx.Parameter(l_labels)
    obj = cvx.Maximize(fmu_i.T*p)

    constraints = []
    for j in range(m_gradients):
        constraints.append(mf.A[j,:,:]*g[j,:].T == mf.B[j,:,:]*p[mf.P[j,:]])
        constraints.append(cvx.norm(g[j,:].T, 2) <= 1)

    prob = cvx.Problem(obj, constraints)

    results = np.zeros(n_image)
    for i in range(n_image):
        fmu_i.value = fmu[:,i]
        prob.solve()
        results[i] = obj.value

    return results
