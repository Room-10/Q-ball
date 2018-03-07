
import numpy as np

from dipy.reconst.shm import real_sph_harm, sph_harm_ind_list as sym_shm_ind_list
from dipy.reconst.shm import order_from_ncoef
from dipy.core.geometry import cart2sphere

def shm_ncoef(sh_order):
    return (sh_order + 1)**2

def sym_shm_ncoef(sh_order):
    return int((sh_order + 2) * (sh_order + 1) // 2)

def shm_mn2k(m, n):
    return 0 if n == 0 else shm_ncoef(n - 1) + (m + n)

def sym_shm_mn2k(m, n):
    return 0 if n == 0 else sym_shm_ncoef(n - 2) + (m + n)

def shm_ind_list(sh_order):
    """
    Returns the degree (n) and order (m) of all the spherical harmonics of
    degree less then or equal to `sh_order`. The results, `m_list`
    and `n_list` are kx1 arrays, where k depends on sh_order. They can be
    passed to :func:`real_sph_harm`.

    Parameters
    ----------
    sh_order : int
        even int > 0, max degree to return

    Returns
    -------
    m_list : array
        orders of even spherical harmonics
    n_list : array
        degrees of even spherical harmonics

    See also
    --------
    real_sph_harm
    """
    n_range = np.arange(0, sh_order + 1, dtype=int)
    n_list = np.repeat(n_range, 2*n_range + 1)

    ncoef = shm_ncoef(sh_order)
    offset = 0
    m_list = np.empty(ncoef, 'int')
    for ii in n_range:
        m_list[offset:offset + 2 * ii + 1] = np.arange(-ii, ii + 1)
        offset = offset + 2 * ii + 1

    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return (m_list, n_list)

def shm_sym2full(sh_order):
    """
    Returns a matrix that converts the representation of a function in symmetric
    spherical harmonics into a representation in full spherical harmonics
    (filling with zeros).

    Returns
    -------
    A : array of shape (ncoef, nsymcoef)
        Matrix where `ncoef = shm_ncoef(sh_order)` and
        `nsymcoef = sym_shm_ncoef(sh_order)`
    """
    M, N = shm_ind_list(sh_order)
    ncoef = shm_ncoef(sh_order)
    A = np.zeros((shm_ncoef(sh_order), sym_shm_ncoef(sh_order)))

    for k in range(ncoef):
        if N[k] % 2 == 0:
            A[k,sym_shm_mn2k(M[k],N[k])] = 1.0

    return A

def shm_grad(sh_order):
    """
    Returns the linear matrix gradient operator for functions represented in
    spherical harmonics up to order `sh_order`.

    Parameters
    ----------
    sh_order : int
        even int > 0, max degree of represented function

    Returns
    -------
    G : array of shape (2*ncoefnew, ncoef)
        Gradient matrix where `ncoef = shm_ncoef(sh_order)` and
        `ncoefnew = shm_ncoef(sh_order + 1)`

    """
    M, N = shm_ind_list(sh_order)
    ncoef = shm_ncoef(sh_order)
    ncoefnew = shm_ncoef(sh_order + 1)
    G = np.zeros((2*ncoefnew, ncoef))

    for k in range(ncoef):
        # Taken from http://geo.mff.cuni.cz/~lh/phd/cha.pdf, cite:
        #
        #   Ladislav Hanyik: Viscoelastic response of the earth:
        #   Initial-value approach. PhD Thesis (1999), p. 117.
        #
        G[shm_mn2k(M[k],N[k] - 1),k] = -(N[k]+1)*np.sqrt(
            (N[k]**2 - M[k]**2)
            /((2*N[k]+1)*(2*N[k]-1))
        )
        G[shm_mn2k(M[k],N[k] + 1),k] = N[k]*np.sqrt(
            ((N[k]+1)**2 - M[k]**2)
            /((2*N[k]+1)*(2*N[k]+3))
        )

        G[ncoefnew + shm_mn2k(-M[k],N[k]),k] = M[k]

    return G

def sym_shm_grad(sh_order):
    """
    Returns the linear matrix gradient operator for functions represented in
    symmetric spherical harmonics up to order `sh_order`.

    Parameters
    ----------
    sh_order : int
        even int > 0, max degree of represented function

    Returns
    -------
    G : array of shape (2*ncoefnew, nsymcoef)
        Gradient matrix where `ncoefnew = shm_ncoef(sh_order + 1)` and
        `nsymcoef = sym_shm_ncoef(sh_order)`
    """
    A = shm_sym2full(sh_order)
    G = shm_grad(sh_order)
    return G.dot(A)

def shm_sample(sh_order, theta, phi):
    theta = np.reshape(theta, [-1, 1])
    phi = np.reshape(phi, [-1, 1])
    M, N = shm_ind_list(sh_order)
    return real_sph_harm(M, N, theta, phi)

def sym_shm_sample(sh_order, theta, phi):
    theta = np.reshape(theta, [-1, 1])
    phi = np.reshape(phi, [-1, 1])
    M, N = sym_shm_ind_list(sh_order)
    return real_sph_harm(M, N, theta, phi)

def sym_shm_sample_grad(sample, gradients):
    ncoef = sample.shape[1]
    m_gradients = gradients.shape[1]
    sh_order = int(order_from_ncoef(ncoef))
    G = sym_shm_grad(sh_order)
    G = G.reshape((2,-1,ncoef))
    x, y, z = gradients
    _, theta, phi = cart2sphere(x, y, z)
    gradsample = shm_sample(sh_order + 1, theta, phi)
    return np.einsum('jl,tlk->jtk', gradsample, G)

"""
# ==============================================================================
# ==================================== Tests ===================================
# ==============================================================================

def checkDerivative(fun, x):
    fx, gradfx = fun(x, True)
    m0 = 1
    N = 8
    for m in range(m0,m0+N):
        h = 10**(-m)
        err = 0
        for i in range(20):
            ve = np.random.randn(x.size)
            ve /= np.sqrt(np.sum(ve**2))
            v = h*ve
            taylor = fx + np.einsum('i,i->', gradfx, v)
            err = max(err, np.abs(fun(x + v) - taylor))
        print('%02d: % 7.2e % 7.2e % 7.2e' % (m, h, err, err/h**2))

# theta \in [0, 2*pi], phi \in [0, pi], |m| <= n
# real_sph_harm(M, N, phi, theta)

theta_test = 0.6*np.pi
phi_test = 0.4*np.pi

sh_order_test = 6
G_test = shm_grad(sh_order_test)

ncoef_test = shm_ncoef(sh_order_test)
#coef_test = np.random.randn(ncoef_test,1)
coef_test = np.zeros((ncoef_test,1))
coef_test[shm_mn2k(-3,6),0] = 1.0

def test_f(x, grad=False):
    theta, phi = x[:]

    fx = shm_sample(sh_order_test, theta, phi).dot(coef_test)
    fx = fx.ravel()[0]

    if grad:
        grad_coef = G_test.dot(coef_test).reshape((2,-1)).T
        gradfx = shm_sample(sh_order_test+1, theta, phi).dot(grad_coef)
        gradfx = gradfx.ravel()
        gradfx[0] *= 1/np.sin(phi)
        return fx, gradfx
    else:
        return fx

def test_f1(x, grad=False):
    result = test_f(np.array([x[0],phi_test]), grad)
    if grad:
        return result[0], result[1][0:1]
    else:
        return result

def test_f2(x, grad=False):
    result = test_f(np.array([theta_test,x[0]]), grad)
    if grad:
        return result[0], result[1][1:2]
    else:
        return result

print("Full function:")
checkDerivative(test_f, np.array([theta_test, phi_test]))
print("First component:")
checkDerivative(test_f1, np.array([theta_test]))
print("Second component:")
checkDerivative(test_f2, np.array([phi_test]))
"""
