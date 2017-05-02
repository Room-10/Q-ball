
import numpy as np
import logging

def manifold_mean(expmap, vecs, weights, max_iter=200):
    """ Calculate arithmetic (geodesic) means of points.

    Args:
        expmap : dictionary with entries 'inv' and 'map' denoting functions
                 inv(location, pfrom, vto) and map(location, vfrom) that project
                 from/into the tangent space at point `location`
        vecs : numpy array of shape (l_vecs, s_manifold). The rows of `vecs`
               specify points on the manifold.
        weights : numpy array of shape (n_means, l_vecs)
        max_iter : Number of iterations.
    Returns:
        numpy array of shape (n_means, s_manifold)
    """

    l_vecs, s_manifold = vecs.shape
    n_means = weights.shape[0]
    assert(weights.shape[1] == l_vecs)

    result = np.zeros((n_means, s_manifold))
    tang = np.zeros(s_manifold)
    vecs_inv = 0*vecs

    if n_means*l_vecs > 100:
        # Output only if it will take a bit longer...
        logging.info("Computing {n_means} means of {l_vecs} points in "\
              "at most {maxiter} steps...".format(
                  maxiter=max_iter,
                  n_means=n_means,
                  l_vecs=l_vecs))

    for i, (res, w_vec) in enumerate(zip(result, weights)):
        # take vector associated with maximum weight as starting point
        cur = vecs[w_vec.argmax()]

        w_sum_inv = 1.0/np.einsum('i->', w_vec)
        for _iter in range(max_iter):
            expmap['inv'](cur, vecs, vecs_inv)
            # tang is the mean on the tangent space
            np.einsum('i,ij->j', w_vec, vecs_inv, out=tang)
            tang *= w_sum_inv
            cur = expmap['map'](cur, tang)
        res[:] = cur

    return result