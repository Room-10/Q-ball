
import numpy as np
from numpy.linalg import norm

from numba import jit

def project_gradients(g, lbd, g_norms):
    """ Project the gradients g to the spectral ball of radius lbd.

    Args:
        g : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
            where `d_image in (1,2) or s_manifold in (1,2)`. The g^ij are
            projected to the ball with radius lbd.
        lbd : radius of the spectral ball
        g_norms : numpy array of shape (n_image, m_gradients)
                  for caching purposes
    Returns:
        nothing, the result is stored in place!
    """
    n_image, m_gradients, s_manifold, d_image = g.shape

    if d_image == 1 or s_manifold == 1:
        # L2 projection (wrt. Frobenius norm)
        norms_spectral(g, g_norms)
        np.fmax(lbd, g_norms, out=g_norms)
        np.divide(lbd, g_norms, out=g_norms)
        g[:] = np.einsum('ij,ij...->ij...', g_norms, g)
    elif d_image == 2 or s_manifold == 2:
        # spectral projection (wrt. spectral norm)
        spectral_projection_2d(g, lbd)
    else:
        raise Exception("Dimension error: d_image={:d}, s_manifold={:d}" \
                        .format(d_image, s_manifold))

@jit
def spectral_projection_2d(g, lbd):
    """ Project the gradients g to the spectral ball of radius lbd

    Args:
        g : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
            where `d_image in (1,2) or s_manifold in (1,2)`. The g^ij are
            projected to the ball with radius lbd.
        lbd : radius of the spectral ball
    Returns:
        nothing, the result is stored in place!
    """
    n_image, m_gradients, s_manifold, d_image = g.shape
    V = np.empty((2,2))
    C = np.empty((2,2))
    S = np.zeros((2,2))
    for i in range(n_image):
        for j in range(m_gradients):
            if d_image == 2:
                A = g[i,j]
            else:
                A = g[i,j].T
            np.dot(A.T, A, out=C)

            # Compute eigenvalues
            trace = C[0,0] + C[1,1]
            d = C[0,0]*C[1,1] - C[0,1]*C[0,1]
            d = np.sqrt(max(0.0, 0.25*trace**2 - d))
            lmin, lmax = max(0.0, 0.5*trace - d), max(0.0, 0.5*trace + d)
            smin, smax = np.sqrt(lmin), np.sqrt(lmax)

            if smax > lbd:
                # Compute orthonormal eigenvectors
                if C[0,1] == 0.0:
                    if C[0,0] >= C[1,1]:
                        V[0,1] = V[1,0] = 0.0
                        V[0,0] = V[1,1] = 1.0
                    else:
                        V[0,1] = V[1,0] = 1.0
                        V[0,0] = V[1,1] = 0.0
                else:
                    V[0,0] = V[0,1] = C[0,1]
                    V[1,0] = lmax - C[0,0]
                    V[1,1] = lmin - C[0,0]
                    Vnorm = np.sqrt(V[0,0]**2 + V[1,0]**2)
                    V[0,0] /= Vnorm
                    V[1,0] /= Vnorm
                    Vnorm = np.sqrt(V[0,1]**2 + V[1,1]**2)
                    V[0,1] /= Vnorm
                    V[1,1] /= Vnorm

                # Thresholding of eigenvalues
                S[0,0] = min(smax, lbd)/smax
                S[1,1] = min(smin, lbd)/smin if smin > 0.0 else 0.0

                # proj(A) = A * V * S * V^T
                A[:] = np.dot(A, V.dot(S).dot(V.T))

def norms_spectral(g, res):
    """ Compute the spectral norm of each g^ij and store it in res.

    Args:
        g : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
            where `d_image in (1,2) or s_manifold in (1,2)`
        res : numpy array of shape (n_image, m_gradients)
    Returns:
        nothing, the result is written to res
    """
    n_image, m_gradients, s_manifold, d_image = g.shape

    if d_image == 1 or s_manifold == 1:
        np.einsum('ijlm,ijlm->ij', g, g, out=res)
        np.sqrt(res, out=res)
    elif d_image == 2 or s_manifold == 2:
        res[:] = norm(g, ord=2, axis=(2,3))
    else:
        raise Exception("Dimension error: d_image={:d}, s_manifold={:d}" \
                        .format(d_image, s_manifold))

def norms_nuclear(g, res):
    """ Compute the nuclear norm of each g^ij and store it in res.

    The nuclear norm is the dual norm of the spectral norm (see above)

    Args:
        g : numpy array of shape (n_image, m_gradients, s_manifold, d_image)
            where `d_image in (1,2) or s_manifold in (1,2)`
        res : numpy array of shape (n_image, m_gradients)
    Returns:
        nothing, the result is written to res
    """
    n_image, m_gradients, s_manifold, d_image = g.shape

    if d_image == 1 or s_manifold == 1:
        np.einsum('ijlm,ijlm->ij', g, g, out=res)
        np.sqrt(res, out=res)
    elif d_image == 2 or s_manifold == 2:
        res[:] = norm(g, ord='nuc', axis=(2,3))
    else:
        raise Exception("Dimension error: d_image={:d}, s_manifold={:d}" \
                        .format(d_image, s_manifold))

def project_duals(p, lbd, p_norms):
    """ Project the p^i to the Frobenius ball of radius lbd.

    Args:
        g : numpy array of shape (l_shm, d_image, n_image)
            The p^i are projected to the ball with radius lbd.
        lbd : radius of the ball
        p_norms : numpy array of shape (n_image,)
                  for caching purposes
    Returns:
        nothing, the result is stored in place!
    """
    # L2 projection (wrt. Frobenius norm)
    norms_frobenius(p, p_norms)
    np.fmax(lbd, p_norms, out=p_norms)
    np.divide(lbd, p_norms, out=p_norms)
    p[:] = np.einsum('i,...i->...i', p_norms, p)

def norms_frobenius(p, res):
    """ Compute the frobenius norm of each p^i and store it in res.

    Args:
        p : numpy array of shape (l_shm, d_image, n_image)
        res : numpy array of shape (n_image,)
    Returns:
        nothing, the result is written to res
    """
    np.einsum('kti,kti->i', p, p, out=res)
    np.sqrt(res, out=res)
