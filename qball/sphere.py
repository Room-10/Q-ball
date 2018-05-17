
from __future__ import division

from qball.tools.plot import plot_mesh3
from qball.tools.norm import normalize
from qball.tools.mean import manifold_mean
from qball.util import output_dir_create

import pickle
import numpy as np
from scipy.spatial import SphericalVoronoi

import logging

def load_sphere(vecs=None, refinement=2):
    if vecs is not None:
        sphere_lvl = vecs.shape[1]
    else:
        sphere_lvl = "r{}".format(refinement)
    sphere_file = "cache/sphere-{}.pickle".format(sphere_lvl)
    try:
        sph = pickle.load(open(sphere_file, 'rb'))
    except:
        print("No cached sphere({}). Preparing...".format(sphere_lvl))
        output_dir_create("cache")
        sph = Sphere(vecs=vecs, refinement=refinement)
        pickle.dump(sph, open(sphere_file, 'wb'))
    return sph

class Sphere(object):
    """A 2D sphere, parametrized directly by the embedding in R^3."""

    def __init__(self, refinement=2, vecs=None):
        """ Setup a triangular grid on the 2-sphere.

        Args:
            vecs : array of shape (3, l_labels)
            refinement : if vecs is None, the triangulation is generated using
                         trisphere(refinement), defined below
        """
        if vecs is None:
            vecs, tris = trisphere(refinement)
            vecs = normalize(vecs)
        else:
            assert(vecs.shape[0] == 3)
            vecs = normalize(vecs)
            sv = SphericalVoronoi(vecs.T)
            sv.sort_vertices_of_regions()
            tris = sv._tri.simplices.T
        assert(tris.shape[0] == 3)

        s_manifold = 2
        m_gradients = tris.shape[1]
        r_points = 3
        l_labels = vecs.shape[1]

        logging.info("Set up for 2-sphere with {l} labels and {m} gradients." \
            .format(s=s_manifold, l=l_labels,m=m_gradients))

        self.v = vecs
        self.faces = tris
        self.mdims = {
            's_manifold': s_manifold,
            'm_gradients': m_gradients,
            'r_points': r_points,
            'l_labels': l_labels
        }

        self.v_grad = np.zeros((3, m_gradients), order='C')
        for j in range(m_gradients):
            # v : columns correspond to the vertices of triangle j
            v = self.v[:,self.faces[:,j]]
            # v0 : centroid of triangle j
            # taking the mean is a bottle neck here!
            self.v_grad[:,j] = self.mean(v).ravel()

        # b { l_labels }
        # P { m_gradients, r_points }
        # b : vol. elements, s.t. sum(self.b) converges to 4*PI as n to inf
        self.b = np.zeros((l_labels,), order='C')
        self.b[:] = sphere_areas(vecs)
        self.b_precond = 1.0/np.sqrt(np.einsum('i,i->', self.b, self.b))
        self.P = np.zeros((m_gradients, r_points), order='C', dtype=np.int64)
        self.P[:] = tris.T

        # A { m_gradients, s_manifold, s_manifold }
        # B { m_gradients, s_manifold, r_points }
        self.A = np.zeros((m_gradients, s_manifold, s_manifold), order='C')
        self.B = np.zeros((m_gradients, s_manifold, r_points), order='C')

        self.setup_taylor_grad()

    def setup_taylor_grad(self):
        s_manifold = self.mdims['s_manifold']
        m_gradients = self.mdims['m_gradients']
        r_points = self.mdims['r_points']
        l_labels = self.mdims['l_labels']

        E = np.eye(r_points) - 1.0/r_points * np.ones((r_points, r_points))
        for j in range(m_gradients):
            # v : columns correspond to the vertices of triangle j
            v = self.v[:,self.faces[:,j]]
            # v0 : centroid of triangle j
            v0 = self.v_grad[:,j]

            # D : each column is a tangent vector at v0
            # project the vertices to the tangent space at v0
            D = v.T - v0
            D -= np.einsum('i,j->ij', np.einsum('ij,j->i', D, v0), v0)
            D = normalize(D.T)
            for k in range(D.shape[1]):
                D[:,k] *= self.dist(v[:,k], v0)

            # C : basis of tangent space at v0, orthonormal wrt. euclidean metric
            c1 = normalize(D[:,0]).ravel()
            c2 = normalize(D[:,1] - D[:,1].dot(c1) * c1).ravel()
            C = np.vstack((c1, c2)).T

            # M { r_points, s_manifold }
            M = D.T.dot(C)

            self.A[j] = M.T.dot(E).dot(M)
            self.B[j] = M.T.dot(E)

    def setup_barycentric_grad(self):
        """ This alternative gradient computation is almost identical,
            but it's supposed to give better sublabel-accuracy.

        Testing code:
        >>> import numpy as np
        >>> from qball.sphere import Sphere
        >>> from qball.tools.w1dist import w1_dist
        >>> from qball.tools import normalize_odf
        >>> n = 2
        >>> mf = Sphere(n)
        >>> phi, psi = 18, 33
        >>> f_raw = np.asarray([np.sin(phi*np.pi/180.0), np.cos(phi*np.pi/180.0), 0])
        >>> u_raw = np.asarray([np.sin(psi*np.pi/180.0), np.cos(psi*np.pi/180.0), 0])
        >>> f = np.zeros((mf.mdims['l_labels'], 1), order='C')
        >>> u = f.copy()
        >>> f[:,0] = mf.embed_barycentric(f_raw)
        >>> u[:,0] = mf.embed_barycentric(u_raw)
        >>> normalize_odf(f, mf.b)
        >>> normalize_odf(u, mf.b)
        >>> w1d_1 = w1_dist(f, u, mf)[0]
        >>> mf.setup_barycentric_grad()
        >>> w1d_2 = w1_dist(f, u, mf)[0]
        >>> mfd = mf.dist(f_raw, u_raw)
        >>> print("mf.dist: {}\nw1_dist_taylor: {}\nw1_dist_barycentric: {}".format(
        >>>     180*mfd/np.pi, 180*w1d_1/np.pi, 180*w1d_2/np.pi
        >>> ))
        """
        s_manifold = self.mdims['s_manifold']
        m_gradients = self.mdims['m_gradients']
        r_points = self.mdims['r_points']
        l_labels = self.mdims['l_labels']

        E = np.vstack((-np.ones(s_manifold), np.eye(s_manifold))).T
        for j in range(m_gradients):
            # v : columns correspond to the vertices of triangle j
            v = self.v[:,self.faces[:,j]]
            # v0 : centroid of triangle j
            v0 = self.v_grad[:,j]

            # D : each row is a tangent vector at v0 pointing to a vertex
            D = np.zeros((3, 3))
            expmap_sphere_inv(v0, v.T, D)

            # scal : scalar products <v0, v_i>
            scal = np.einsum('ji,j->i', v, v0)

            # R : d(v0, v_i)
            R = np.arccos(scal)

            # dnR : normal derivatives of the functions v -> d(v,v_i) at v=v0
            dnR = scal/np.sqrt(1 - scal*scal)

            # N : each row is a unit tangent vector at v0 pointing to a vertex
            N = normalize(D.T).T

            # A : matrix for computing barycentric coordinates of v0
            A = np.linalg.inv(np.vstack((D[1] - D[0], D[2] - D[0], v0)).T)[0:-1,:]

            # alph : barycentric coordinates
            alph = np.hstack((0, A.dot(-D[0])))
            alph[0] = 1-sum(alph[1:])

            # Nii : the tensor products N[i]xN[i]
            Nii = [np.einsum('j,k->jk', N[i], N[i]) for i in range(r_points)]

            # HR : the Hessians of the functions v -> -0.5 * d(v,v_i)^2
            HR = [-Nii[i] + R[i]*dnR[i]*(Nii[i] - np.eye(3)) for i in range(r_points)]

            # Dalph: derivative of barycentric coordinate function
            Dalph = A.dot(sum([alph[k]*HR[k] for k in range(s_manifold+1)]))

            # C : orthonormal basis transformation for tangent space at v0
            c1 = normalize(D[0]).ravel()
            c2 = normalize(D[1] - D[1].dot(c1) * c1).ravel()
            C = np.linalg.inv(np.vstack((c1, c2, v0)).T)[0:-1,:]

            self.A[j] = np.eye(s_manifold)
            self.B[j] = C.dot(Dalph.T).dot(E)

    def interpolate(self, f, x):
        return np.einsum('ji,i->j', f, self.embed_barycentric(x))

    def embed_barycentric(self, x):
        """ Embed a 3D vector into the triangulation using barycentric coordinates.

        Args:
            x : the vector to be embedded, numpy array of shape (3,)
        Returns:
            numpy array of shape (l_labels,)

        Test code:
            n = 2
            mf = Sphere(n)
            for theta, phi in zip([23,42,76,155,143,103], [345,220,174,20,18,87]):
                f_in = np.asarray([
                    np.sin(theta*np.pi/180.0)*np.cos(phi*np.pi/180.0),
                    np.sin(theta*np.pi/180.0)*np.sin(phi*np.pi/180.0),
                    np.cos(theta*np.pi/180.0)
                ])
                f_out = mf.mean(mf.v, mf.embed_barycentric(f_in)[:,np.newaxis]).ravel()
                err = np.sqrt(sum((f_in-f_out)**2))
                print("theta: {}, phi: {}, err: {}".format(theta, phi, err))
                assert err <= 1e-15
        """

        xn = normalize(x)[:,0]
        target_triangle = None
        for tri in self.faces.T:
            p1 = self.v[:,tri[0]]
            p2 = self.v[:,tri[1]]
            p3 = self.v[:,tri[2]]
            if tri_contains(xn, p1, p2, p3):
                target_triangle = tri.copy()
                break
        assert target_triangle is not None

        tvecs = np.zeros((3, 3))
        expmap_sphere_inv(xn, np.vstack((p1, p2, p3)), tvecs)
        v1 = tvecs[1] - tvecs[0]
        v2 = tvecs[2] - tvecs[0]
        coords = np.linalg.solve(np.vstack((v1, v2, xn)).T, -tvecs[0])
        assert np.abs(coords[2]) < 1.e-11
        assert coords[0] >= 0 and coords[1] >= 0 and coords[0] + coords[1] <= 1.0

        coords = np.array([1 - coords[0] - coords[1], coords[0], coords[1]])
        weights = np.zeros((self.mdims['l_labels'],), order='C')
        weights[target_triangle] = coords.T
        return weights

    def embed(self, x):
        """ No embedding necessary since already parametrized in R^3 """
        return x

    def plot(self, ax):
        """ Plots this sphere on the given instance `ax` of Axes3d.

        Args:
            ax : Instance of mpl_toolkits.mplot3d.Axes3d

        Test code:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            from qball.sphere import load_sphere
            load_sphere(refinement=2).plot(ax)
            plt.show()
        """
        plot_mesh3(ax, self.v, self.P.T)

    def dist(self, x, y):
        """ Compute geodesic distance of points x and y on the sphere.

        Args:
            x, y : numpy arrays, points on the sphere
        """
        return np.arccos(np.clip(np.einsum('i,i->', x, y), -1.0, 1.0))

    def mean(self, vecs, weights=None):
        """ Calculate arithmetic (geodesic) means of points on the sphere.

        Args:
            vecs : numpy array of shape (s_manifold, l_vecs). The columns
                   specify points on the sphere
            weights : numpy array of shape (l_vecs, n_means).
                      If ommitted, a uniform weight is assumed.
        Returns:
            numpy array of shape (s_manifold, n_means)
        """
        if weights is None:
            weights = np.ones((vecs.shape[1], 1), dtype=np.float64) / vecs.shape[1]

        expmap_sphere = {
            'map': expmap_sphere_map,
            'inv': expmap_sphere_inv
        }
        return manifold_mean(expmap_sphere, vecs.T, weights.T).T

def expmap_sphere_inv(location, pfrom, vto):
    """ exp_l^{-1}(p) = d(l,p) * (p - <p,l>l)/|p - <p,l>l|
    and we use |p - <p,l>l| = sqrt(1 - <p,l>^2) (by |p|=|l|=1)
    as well as d(l,p) = arccos(<p,l>)
    """
    # yxi : <p,l>
    # clamp to avoid problems with arccos
    # (should not lie outside due to normalization)
    yxi = np.clip(np.einsum('ij,j->i', pfrom, location), -1.0, 1.0)

    # skip opposite points, this causes numerical problems
    ind = yxi >= -1.0 + 16*np.spacing(1)
    vto[np.logical_not(ind)] = 0.0
    # factor : d(l,p)/|p - <p,l>l| = arccos(<p,l>)/sqrt(1 - <p,l>^2)
    factor = np.arccos(yxi[ind])/np.fmax(np.spacing(1),np.sqrt(1 - yxi[ind]**2))
    # vto : factor * (p - <p,l>l)
    vto[ind,:] = np.einsum('i,ij->ij', factor,
        pfrom[ind,:] - np.einsum('i,j->ij', yxi[ind], location)
    )

def expmap_sphere_map(location, vfrom):
    """ exp_l(v) = cos(|v|) * l  + sin(|v|) * v/|v| """
    c = np.sqrt(np.einsum('i,i->', vfrom, vfrom))
    pto = np.cos(c)*location + np.sin(c)/max(np.spacing(1),c)*vfrom
    # normalizing prevents errors from accumulating
    return normalize(pto).reshape(-1)

def tri_contains(x, p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = -x
    coeff = np.vstack((v1, v2, v3)).T
    try:
        coords = np.linalg.solve(coeff, -p1)
        if not coords[2] > 0: return False
        if not all([coords[0] >= 0, coords[1] >= 0]): return False
        if not coords[0] + coords[1] <= 1.0: return False
        return True
    except np.linalg.linalg.LinAlgError:
        return False

def trisphere(n):
    """ Calculates a sphere triangulation of 12*(4^n) vertices.

    The algorithm starts from an icosahedron (12 vertices, 20 triangles) and
    applies the following procedure `n` times: Each triangular face is split
    into four triangles using the triangle's edge centers as new vertices.

    Args:
        n : grade of refinement; n=0 means 12 vertices (icosahedron).
    Returns:
        tuple (verts, tris)
        verts : numpy array, each column corresponds to a point on the sphere
        tris : numpy array, each column defines a triangle on the sphere
               through indices into `verts`
    """
    # Regular icosahedron:
    # (X, Z) : solution to X/Z = Z/(X + Z) and 1 = X^2 + Z^2
    X = ((5 - 5**0.5)/10)**0.5
    Z = (1 - X**2)**0.5

    # verts { 3, 12 }
    verts = np.array([
        [ -X, 0.0,   Z],
        [  X, 0.0,   Z],
        [ -X, 0.0,  -Z],
        [  X, 0.0,  -Z],
        [0.0,   Z,   X],
        [0.0,   Z,  -X],
        [0.0,  -Z,   X],
        [0.0,  -Z,  -X],
        [  Z,   X, 0.0],
        [ -Z,   X, 0.0],
        [  Z,  -X, 0.0],
        [ -Z,  -X, 0.0]
    ]).T

    # tris { 3, 20 }
    tris = np.array([
        [ 0,  4,  1],
        [ 0,  9,  4],
        [ 9,  5,  4],
        [ 4,  5,  8],
        [ 4,  8,  1],
        [ 8, 10,  1],
        [ 8,  3, 10],
        [ 5,  3,  8],
        [ 5,  2,  3],
        [ 2,  7,  3],
        [ 7, 10,  3],
        [ 7,  6, 10],
        [ 7, 11,  6],
        [11,  0,  6],
        [ 0,  1,  6],
        [ 6,  1, 10],
        [ 9,  0, 11],
        [ 9, 11,  2],
        [ 9,  2,  5],
        [ 7,  2, 11]
    ]).T
    # t = np.array([[0, 1, 4]]).T;

    for i in range(n):
        vn = verts

        # numv : current number of verts
        numv = vn.shape[1]

        # edgecenters : symmetric matrix of indices into `vn`
        edgecenters = np.zeros([verts.shape[1]]*2, dtype=np.int64)

        # iterate over triangles, adding edge centers to `vn` and
        # making note of it in `edgecenters`
        for tri in tris.T:
            # the if clauses make sure we don't get duplicates
            if tri[0] < tri[1]:
                vn = np.hstack((vn,
                    normalize(0.5*(verts[:,tri[0]] + verts[:,tri[1]]))
                ))
                edgecenters[tri[0],tri[1]] = numv
                edgecenters[tri[1],tri[0]] = numv
                numv += 1
            if tri[1] < tri[2]:
                vn = np.hstack((vn,
                    normalize(0.5*(verts[:,tri[1]] + verts[:,tri[2]]))
                ))
                edgecenters[tri[1],tri[2]] = numv
                edgecenters[tri[2],tri[1]] = numv
                numv += 1
            if tri[2] < tri[0]:
                vn = np.hstack((vn,
                    normalize(0.5*(verts[:,tri[2]] + verts[:,tri[0]]))
                ))
                edgecenters[tri[2],tri[0]] = numv
                edgecenters[tri[0],tri[2]] = numv
                numv += 1

        # tn { 3, 4*tris.shape[1] }
        # build up as list then convert to numpy array
        tn = []

        # each triangle is split into four using the edge centers
        for tri in tris.T:
            a = edgecenters[tri[1],tri[2]]
            b = edgecenters[tri[2],tri[0]]
            c = edgecenters[tri[0],tri[1]]
            if a == 0 or b == 0 or c == 0:
                print("Damn")
            tn.extend([
                [tri[0], c, b],
                [tri[1], a, c],
                [tri[2], b, a],
                [        a, b, c]
            ])
        tn = np.asarray(tn).T

        verts = vn
        tris = tn
    return verts, tris

def sphere_areas(v):
    """ Calculates areas of Voronoi regions corresponding to given points.

    Args:
        v : numpy array, each column corresponds to a point on the sphere
    Returns:
        1d numpy array, areas of the Voronoi regions corresponding to each of the points
    """
    areas = np.zeros(v.shape[1])
    sv = SphericalVoronoi(v.T)
    # sort the vertices for poly_area()
    sv.sort_vertices_of_regions()
    """
    # (optional) for visualization of the voronoi regions:
    from matplotlib import colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot generator points
    ax.scatter(v[0], v[1], v[2], c='b')
    # plot Voronoi vertices
    ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], c='g')
    # indicate Voronoi regions (as Euclidean polygons)
    for region in sv.regions:
       random_color = colors.rgb2hex(np.random.rand(3))
       polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
       polygon.set_color(random_color)
       ax.add_collection3d(polygon)
    plt.show()
    """
    for j, region in enumerate(sv.regions):
        areas[j] = spherical_poly_area(sv.vertices[region])
    return areas

def spherical_poly_area(poly):
    """ Calculate area of a given polygon on the 2-sphere.

    The algorithm splits the polygon into triangles and uses Girard's formula

    Args:
        poly : numpy array, rows correspond to vertices of the polygon.
               Vertices are expected to be in either clockwise or
               counterclockwise order!
    """
    if poly.shape[0] < 3:
        # degenerate case
        return 0
    result = 0.0
    N = poly.shape[0]
    for k in range(1,N-1):
        result += spherical_tri_area(np.vstack((poly[0], poly[k], poly[k+1])))
    return result

def spherical_tri_area(tri):
    """ Calculate area of a given triangle on the 2-sphere.

    The algorithm computes the lengths of the triangle's sides. Spherical
    trigonometry then allows to compute the angles which can be plugged into
    Girard's formula.

    Args:
        tri : 3x3 numpy array, each row is a vertex of the triangle.
    """
    alen = np.arccos(tri[1].dot(tri[2]))
    blen = np.arccos(tri[2].dot(tri[0]))
    clen = np.arccos(tri[0].dot(tri[1]))

    s = 0.5*(alen + blen + clen)
    sin_s_a = np.sin(s - alen)
    sin_s_b = np.sin(s - blen)
    sin_s_c = np.sin(s - clen)
    k = (sin_s_a*sin_s_b*sin_s_c/np.sin(s))**0.5

    a = 2.0*np.arctan(k/sin_s_a)
    b = 2.0*np.arctan(k/sin_s_b)
    c = 2.0*np.arctan(k/sin_s_c)
    return a + b + c - np.pi
