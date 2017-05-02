
import numpy as np
from scipy.special import sph_harm, lpmv

def coords_cartesian(r_phi_theta):
    """ Convert spherical coordinates to cartesian coordinates

    Args:
        r_phi_theta : numpy array of shape (3,), (r, phi, theta) such that
                      phi is azimuth [0, 2pi) and theta is inclination [0, pi].
    Returns:
        numpy array of shape (3,) containing xyz coordinates
    """
    r, phi, theta = r_phi_theta
    return np.array([
        r*np.sin(theta)*np.cos(phi),
        r*np.sin(theta)*np.sin(phi),
        r*np.cos(theta)
    ])

def coords_spherical(xyz):
    """ Convert cartesian coordinates to spherical coordinates

    Args:
        xyz : numpy array of shape (3,) containing xyz coordinates
    Returns:
        numpy array of shape (3,) containing (r, phi, theta) such that
        phi is azimuth [0, 2pi) and theta is inclination [0, pi].
    """
    xy = xyz[0]**2 + xyz[1]**2
    r = np.sqrt(xy + xyz[2]**2)
    theta = np.arccos(xyz[2]/r) # inclination
    phi = np.arctan2(xyz[1], xyz[0]) # azimuth
    phi = phi if phi > 0 else phi + 2*np.pi
    return np.array([r, phi, theta])

def coords_shm(vecs, order=5):
    """ Change of coordinates (coc) from spherical harmonics to triangular grid

    Args:
        vecs : numpy array of shape (3,N), columnwise points on the sphere
        order : order of spherical harmonics
    Returns:
        numpy array of shape (N,o*(o+2)) containing coc matrix.
    """
    assert(vecs.shape[0] == 3)

    Y = np.zeros((vecs.shape[1], order*(order + 2)))
    for (j,v) in enumerate(vecs.T):
        nk = 0
        for n in range(order):
            for k in range(-n, n+1):
                Y[j,nk] = sph_harm(k, n, *coords_spherical(v)[1:])
                nk += 1
    return Y

def inverse_frt_lapl(order=5):
    """ Lin. operator for inverse FRT and Laplace in shm coordinates

    Args:
        order : order of spherical harmonics
    Returns:
        numpy array of shape (o*(o+2),) containing diagonal elements
    """
    nk = 0
    A = np.zeros((order*(order + 2),))
    for n in range(order):
        for k in range(-n, n+1):
            lbd = -n*(n+1)
            P0 = lpmv(0, n, 0)
            FRT_lbd = 2*np.pi*P0
            A[nk] = 1.0/(lbd*FRT_lbd)
            nk += 1
    return A