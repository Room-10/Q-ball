
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import numba

def truncate(x, n):
    k = -int(np.floor(np.log10(abs(x))))
    # Example: x = 0.006142 => k = 3 / x = 2341.2 => k = -3
    k += n - 1
    if k > 0:
        x_str = str(abs(x))[:(k+2)]
    else:
        x_str = str(abs(x))[:n]+"0"*(-k)
    return np.sign(x)*float(x_str)

def normalize_odf(odf, vol):
    odf_flat = odf.reshape(odf.shape[0], -1)
    odf_sum = np.einsum('k,ki->i', vol, odf_flat)
    odf_flat[:] = np.einsum('i,ki->ki', 1.0/odf_sum, odf_flat)

def plot_mesh3(ax, vecs, tris):
    """ Plots a surface according to a given triangulation.

    Args:
        ax : Instance of mpl_toolkits.mplot3d.Axes3d
        vecs : numpy array of shape (3, l_vecs) containing the grid points of the
               surface
        tris : numpy array of shape (3, l_vecs). Each column contains the three
               indices (wrt. `vecs`) of a triangle's vertices.

    Test code:
        import matplotlib.pyplot as plt
        fig = plt.figure(facecolor="white")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0.1)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlim((-1.1,1.1))
        ax.set_ylim((-1.1,1.1))
        ax.set_zlim((-1.1,1.1))
        ax.view_init(azim=30)
        ax.scatter(1.01*sv.points[:,0], 1.01*sv.points[:,1], 1.01*sv.points[:,2])
        plot_mesh3(ax, sv.points.T, sv._tri.simplices.T)
        plt.show()
    """
    vx = vecs[0]
    vy = vecs[1]
    vz = vecs[2]

    tmp = np.tile(tris, (2,1))

    verts = np.dstack((vx[tmp].T, vy[tmp].T, vz[tmp].T))
    collection = Poly3DCollection(verts, linewidths=0.3, alpha=0.8)
    collection.set_facecolor([1,1,1])
    collection.set_edgecolor([0.5,0.5,0.5])
    ax.add_collection3d(collection)
    #ax.add_collection3d(
    #    Line3DCollection(verts, colors='k', linewidths=0.2, linestyle=':')
    #)
    #for k in range(vecs.shape[1]):
    #    ax.text(1.1*vecs[0,k], 1.1*vecs[1,k], 1.1*vecs[2,k], str(k))

@numba.njit
def apply_PB(pgrad, P, B, w, precond=False, inpaint_nloc=np.empty(0, dtype=np.bool)):
    """ Does this: pgrad[P] -= np.einsum('jlm,ijlt->jmti', B, w)
    Unfortunately, advanced indexing without creating a copy is impossible.
    """
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            for l in range(w.shape[2]):
                for m in range(B.shape[2]):
                    for t in range(w.shape[3]):
                        if inpaint_nloc.size == 0 or inpaint_nloc[i]:
                            if precond:
                                pgrad[P[j,m],t,i] += np.abs(B[j,l,m])
                            else:
                                pgrad[P[j,m],t,i] -= B[j,l,m]*w[i,j,l,t]
