
import numpy as np
from dipy.core.gradients import GradientTable
from dipy.sims.voxel import single_tensor

from manifold_sphere import load_sphere
from scipy.stats import rice

def rotation_around_axis(v, theta):
    """Return the matrix that rotates 3D data an angle of theta around
    the axis v.
    Parameters
    ----------
    v : (3,) ndarray
        Axis of rotation.
    theta : float
        Angle of rotation in radians.
    References
    ----------
    http://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
    """
    v = np.asarray(v)
    if not v.size == 3:
        raise ValueError("Axis of rotation should be 3D vector.")
    if not np.isscalar(theta):
        raise ValueError("Angle of rotation must be scalar.")

    v = v / np.linalg.norm(v)
    C = np.array([[ 0,   -v[2], v[1]],
                  [ v[2], 0,   -v[0]],
                  [-v[1], v[0], 0]])

    return np.eye(3) + np.sin(theta) * C + (1 - np.cos(theta)) * C.dot(C)

diffusion_evals = np.array([1800e-6, 200e-6, 200e-6]) # mm^2/s

def two_fiber_signal(gtab, angle, w=[0.5, 0.5], SNR=None):
    angle = angle / 2.
    R0 = rotation_around_axis([0, 0, 1], np.deg2rad(-angle))
    R1 = rotation_around_axis([0, 0, 1], np.deg2rad(angle))
    E = w[0] * single_tensor(gtab, S0=1, evecs=R0, evals=diffusion_evals, snr=SNR)
    E += w[1] * single_tensor(gtab, S0=1, evecs=R1, evals=diffusion_evals, snr=SNR)
    return E

def one_fiber_signal(gtab, angle, SNR=None):
    R = rotation_around_axis([0, 0, 1], np.deg2rad(angle))
    E = single_tensor(gtab, S0=1, evecs=R, evals=diffusion_evals, snr=SNR)
    return E

def synthetic(bval=3000):
    imagedims = (8,)
    d_image = len(imagedims)
    n_image = np.prod(imagedims)

    sph = load_sphere(refinement=2)
    l_labels = sph.mdims['l_labels']
    gtab = GradientTable(bval * sph.v.T, b0_threshold=0)

    S_data = np.stack([
        one_fiber_signal(gtab, 0+r)
        for r in 10*np.random.randn(n_image)
    ]).reshape(imagedims + (l_labels,))
    
    # add noise
    np.random.seed(seed=234234)
    sigma = 0.5   # sigma=0.5, v=0.1 is a good choice
    v = 0.1
#    mean, var = rice.stats(v,scale=sigma,moments='mv')
    noise = rice.rvs(v,scale=sigma,size=S_data.size)
    S_data += noise.reshape(S_data.shape)

    return S_data, gtab