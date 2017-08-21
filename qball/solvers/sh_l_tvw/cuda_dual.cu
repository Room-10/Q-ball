
__global__ void linop1(double *x, double *ygrad)
{
    /* ggrad^ij = A^j' w^ij
     *
     * pgrad = 0
     * pgrad_t^i += - P^j' B^j' w_t^ij
     *
     * q0grad = b'u1
     */

    SUBVAR_x_u1(u1,x)
    SUBVAR_x_w(w,x)
    SUBVAR_y_q0(q0grad,ygrad)
    SUBVAR_y_g(ggrad,ygrad)
    SUBVAR_y_p(pgrad,ygrad)

    // global thread index
    int _mj = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int t = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if (_mj >= sm_skip || i >= n_image || t >= d_image)
       return;

    // these have to be computed on the fly
    int m = _mj / m_gradients;
    int j = _mj % m_gradients;

    // iteration variables and misc.
    int ll, mm, jj, kk, idx;
    double newval;

    // ggrad[i,j,m,t]
    idx = i*msd_skip + j*sd_skip + m*d_image + t;
    newval = 0.0;

    // ggrad^ij = A^j' w^ij
    for (ll = 0; ll < s_manifold; ll++) {
        // jlm,ijlt->ijmt
        newval += A[j*ss_skip + ll*s_manifold + m] *
                    w[i*msd_skip + j*sd_skip + ll*d_image + t];
    }
    ggrad[idx] = newval;

    if (_mj == 0) {
        // pgrad = 0
        for (kk = 0; kk < l_labels; kk++) {
            pgrad[kk*nd_skip + t*n_image + i] = 0.0;
        }

        // pgrad_t^i += - P^j' B^j' w_t^ij
        for (jj = 0; jj < m_gradients; jj++) {
            for (mm = 0; mm < r_points; mm++) {
                idx = P[jj*r_points + mm]*nd_skip + t*n_image + i;
                newval = pgrad[idx];
                for (ll = 0; ll < s_manifold; ll++) {
                    // jlm,ijlt->jmti
                    newval -= B[jj*sr_skip + ll*r_points + mm] *
                                w[i*msd_skip + jj*sd_skip + ll*d_image + t];
                }
                pgrad[idx] = newval;
            }
        }

        if (t == 0) {
            // q0grad = b'u1
            newval = 0.0;
            for (kk = 0; kk < l_labels; kk++) {
                newval += b[kk]*u1[kk*n_image + i];
            }
            newval *= b_precond;
            q0grad[i] = newval;
        }
    }
}

__global__ void linop2(double *x, double *ygrad)
{
    /* pgrad += diag(b) D u1 (D is the gradient on a staggered grid)
     *
     * q1grad = Y v - u1
     *
     * q2grad = Y M v - u2
     */

    SUBVAR_x_u1(u1,x)
    SUBVAR_x_u2(u2,x)
    SUBVAR_x_v(v,x)
    SUBVAR_y_p(pgrad,ygrad)
    SUBVAR_y_q1(q1grad,ygrad)
    SUBVAR_y_q2(q2grad,ygrad)

    // global thread index
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int t = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if (k >= l_labels || i >= n_image || t >= d_image)
       return;

    // iteration variable and misc.
    int aa, base, idx;
    double newval, fac;

    // skip points on "bottom right" boundary
    int is_boundary = false;
    int curr_i = i, curr_dim = 0;
    for (aa = d_image - 1; aa >= 0; aa--) {
        curr_dim = curr_i / skips[aa];
        curr_i = curr_i % skips[aa];
        if (curr_dim == imagedims[d_image - 1 - aa] - 1) {
            is_boundary = true;
            break;
        }
    }
    // pgrad[k,t,i]
    idx = k*nd_skip + t*n_image + i;
    newval = pgrad[idx];
    fac = b[k]/(double)navgskips;

    // pgrad += diag(b) D u1 (D is the gradient on a staggered grid)
    if (!is_boundary) {
        for (aa = 0; aa < navgskips; aa++) {
            base = i + avgskips[t*navgskips + aa];
            newval += fac*(
                u1[k*n_image + (base + skips[t])] - u1[k*n_image + base]
            );
        }
    }
    pgrad[idx] = newval;

    if (t == 0) {
        // q1grad[k,i]
        idx = k*n_image + i;
        newval = -u1[idx];

        // q1grad = Y v - u1
        for (aa = 0; aa < l_shm; aa++) {
            newval += Y[k*l_shm + aa]*v[aa*n_image + i];
        }
        q1grad[idx] = newval;

        // q2grad[k,i]
        idx = k*n_image + i;
        newval = -u2[idx];

        // q2grad = Y M v - u2
        for (aa = 0; aa < l_shm; aa++) {
            newval += Y[k*l_shm + aa]*M[aa]*v[aa*n_image + i];
        }
        q2grad[idx] = newval;
    }
}

#ifdef precond
__global__ void prox_dual(double *y, double *ysigma)
#else
__global__ void prox_dual(double *y, double sigma)
#endif
{
    /* This function makes heavy use of registers (34 32-bit registers), so
     * that it will not run with more than 960 threads per block on compute
     * capability 2.x!
     *
     * g = proj(g, lbd)
     * q0 -= sigma
     */

    SUBVAR_y_g(g,y)
    SUBVAR_y_q0(q0,y)

#if (d_image <= s_manifold)
// A := gij, a (d_image x s_manifold)-matrix
#define LIM s_manifold
#define STEP1 d_image
#define STEP2 (1)
#else
// A := gij^T, a (s_manifold x d_image)-matrix
#define LIM d_image
#define STEP1 (1)
#define STEP2 d_image
#endif

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (i >= n_image || j >= m_gradients)
       return;

    // iteration variables and misc.
    int mm;
    double *gij = &g[i*msd_skip + j*sd_skip];
    double norm = 0.0;

    // g = proj(g, lbd)
#if (d_image == 1 || s_manifold == 1)
    for (mm = 0; mm < LIM; mm++) {
        norm += gij[mm*STEP1]*gij[mm*STEP1];
    }

    if (norm > lbd*lbd) {
        norm = lbd/sqrt(norm);
        for (mm = 0; mm < LIM; mm++) {
            gij[mm*STEP1] *= norm;
        }
    }
#elif (d_image == 2 || s_manifold == 2)
    double C11 = 0.0, C12 = 0.0, C22 = 0.0,
           V11 = 0.0, V12 = 0.0, V21 = 0.0, V22 = 0.0,
           M11 = 0.0, M12 = 0.0, M21 = 0.0, M22 = 0.0,
           s1 = 0.0, s2 = 0.0,
           trace, d, lmax, lmin, smax, smin;

    // C = A^T A, a (2 x 2)-matrix
    for (mm = 0; mm < LIM; mm++) {
        C11 += gij[mm*STEP1 + 0*STEP2]*gij[mm*STEP1 + 0*STEP2];
        C12 += gij[mm*STEP1 + 0*STEP2]*gij[mm*STEP1 + 1*STEP2];
        C22 += gij[mm*STEP1 + 1*STEP2]*gij[mm*STEP1 + 1*STEP2];
    }

    // Compute eigenvalues
    trace = C11 + C22;
    d = sqrt(fmax(0.0, 0.25*trace*trace - (C11*C22 - C12*C12)));
    lmax = fmax(0.0, 0.5*trace + d);
    lmin = fmax(0.0, 0.5*trace - d);
    smax = sqrt(lmax);
    smin = sqrt(lmin);

    if (smax > lbd) {
        // Compute orthonormal eigenvectors
        if (C12 == 0.0) {
            if (C11 >= C22) {
                V11 = 1.0; V12 = 0.0;
                V21 = 0.0; V22 = 1.0;
            } else {
                V11 = 0.0; V12 = 1.0;
                V21 = 1.0; V22 = 0.0;
            }
        } else {
            V11 = C12       ; V12 = C12;
            V21 = lmax - C11; V22 = lmin - C11;
            norm = hypot(V11, V21);
            V11 /= norm; V21 /= norm;
            norm = hypot(V12, V22);
            V12 /= norm; V22 /= norm;
        }

        // Thresholding of eigenvalues
        s1 = fmin(smax, lbd)/smax;
        s2 = fmin(smin, lbd);
        s2 = (smin > 0.0) ? s2/smin : 0.0;

        // M = V * diag(s) * V^T
        M11 = s1*V11*V11 + s2*V12*V12;
        M12 = s1*V11*V21 + s2*V12*V22;
        M21 = s1*V21*V11 + s2*V22*V12;
        M22 = s1*V21*V21 + s2*V22*V22;

        // proj(A) = A * M
        for (mm = 0; mm < LIM; mm++) {
            // s1, s2 now used as temp. variables
            s1 = gij[mm*STEP1 + 0*STEP2];
            s2 = gij[mm*STEP1 + 1*STEP2];
            gij[mm*STEP1 + 0*STEP2] = s1*M11 + s2*M21;
            gij[mm*STEP1 + 1*STEP2] = s1*M12 + s2*M22;
        }
    }
#endif

    if (j == 0) {
#ifdef precond
        SUBVAR_y_q0(q0sigma,ysigma)
        q0[i] -= q0sigma[i]*b_precond;
#else
        q0[i] -= sigma*b_precond;
#endif
    }
}
