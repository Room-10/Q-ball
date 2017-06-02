
__global__ void DualKernel1(double *u1k, double *u2k, double *vk, double *wk,
                            double *u1bark, double *u2bark, double *vbark, double *wbark,
                            double *pk, double *gk, double *q0k, double *q1k, double *q2k,
                            double *pkp1, double *gkp1, double *q0kp1, double *q1kp1, double *q2kp1,
                            double *b, double *A, double *B, long *P,
                            double *f, double *Y, double *M, double sigma, double tau, double theta,
                            double lbd, double b_precond,
                            double *constraint_u, unsigned char *uconstrloc)
{
    /* gkp1^ij = A^j' wbark^ij
     * gkp1 = gk + sigma*gkp1
     *
     * pkp1 = 0
     * pkp1_t^i += - P^j' B^j' wbark_t^ij
     *
     * q0kp1 = b'ubark - 1
     * q0kp1 = q0k + sigma*q0kp1
     * q0k = q0kp1
     */

    // global thread index
    int _mj = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int t = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if(_mj >= sm_skip || i >= n_image || t >= d_image)
       return;

    // these have to be computed on the fly
    int m = _mj / m_gradients;
    int j = _mj % m_gradients;

    // iteration variables and misc.
    int ll, mm, jj, kk;
    double newval;

    newval = 0.0;
    for(ll = 0; ll < s_manifold; ll++) {
        // jlm,ijlt->ijmt
        newval += A[j*ss_skip + ll*s_manifold + m] *
                    wbark[i*msd_skip + j*sd_skip + ll*d_image + t];
    }
    newval *= sigma;
    gkp1[i*msd_skip + j*sd_skip + m*d_image + t] =
        gk[i*msd_skip + j*sd_skip + m*d_image + t] + newval;

    if(_mj == 0) {
        for(kk = 0; kk < l_labels; kk++) {
            pkp1[kk*nd_skip + t*n_image + i] = 0.0;
        }
        for(jj = 0; jj < m_gradients; jj++) {
            for(mm = 0; mm < r_points; mm++) {
                newval = pkp1[P[jj*r_points + mm]*nd_skip + t*n_image + i];
                for(ll = 0; ll < s_manifold; ll++) {
                    // jlm,ijlt->jmti
                    newval -= B[jj*sr_skip + ll*r_points + mm] *
                                wbark[i*msd_skip + jj*sd_skip + ll*d_image + t];
                }
                pkp1[P[jj*r_points + mm]*nd_skip + t*n_image + i] = newval;
            }
        }

        if(t == 0) {
            newval = 0.0;
            for(kk = 0; kk < l_labels; kk++) {
                newval += b[kk]*u1bark[kk*n_image + i];
            }
            newval = q0k[i] + sigma*b_precond*(newval - 1.0);
            q0kp1[i] = newval;
            q0k[i] = newval;
        }
    }
}

__global__ void DualKernel2(double *u1k, double *u2k, double *vk, double *wk,
                            double *u1bark, double *u2bark, double *vbark, double *wbark,
                            double *pk, double *gk, double *q0k, double *q1k, double *q2k,
                            double *pkp1, double *gkp1, double *q0kp1, double *q1kp1, double *q2kp1,
                            double *b, double *A, double *B, long *P,
                            double *f, double *Y, double *M, double sigma, double tau, double theta,
                            double lbd, double b_precond,
                            double *constraint_u, unsigned char *uconstrloc)
{
    /* pkp1 += diag(b) D u1bark (D is the gradient on a staggered grid)
     * pkp1 = pk + sigma*pkp1
     * pk = pkp1
     *
     * q1kp1 = Y vbark - u1bark
     * q1kp1 = q1k + sigma*q1kp1
     * q1k = q1kp1
     *
     * q2kp1 = Y M vbark - u2bark
     * q2kp1 = q2k + sigma*q2kp1
     * q2k = q2kp1
     */

    // global thread index
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int t = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if(k >= l_labels || i >= n_image || t >= d_image)
       return;

    // iteration variable and misc.
    int aa, base;
    double newval;

    // skip points on "bottom right" boundary
    int is_boundary = false;
    int curr_i = i, curr_dim = 0;
    for(aa = d_image - 1; aa >= 0; aa--) {
        curr_dim = curr_i / skips[aa];
        curr_i = curr_i % skips[aa];
        if(curr_dim == imagedims[d_image - 1 - aa] - 1) {
            is_boundary = true;
            break;
        }
    }

    newval = pkp1[k*nd_skip + t*n_image + i];
    if(!is_boundary) {
        for(aa = 0; aa < navgskips; aa++) {
            base = i + avgskips[t*navgskips + aa];
            newval += b[k]/(double)navgskips * (
                u1bark[k*n_image + (base + skips[t])] - u1bark[k*n_image + base]
            );
        }
    }

    newval = pk[k*nd_skip + t*n_image + i] + sigma*newval;
    pkp1[k*nd_skip + t*n_image + i] = newval;
    pk[k*nd_skip + t*n_image + i] = newval;

    if(t == 0) {
        newval = -u1bark[k*n_image + i];
        for(aa = 0; aa < l_shm; aa++) {
            newval += Y[k*l_shm + aa]*vbark[aa*n_image + i];
        }
        newval = q1k[k*n_image + i] + sigma*newval;
        q1kp1[k*n_image + i] = newval;
        q1k[k*n_image + i] = newval;

        newval = -u2bark[k*n_image + i];
        for(aa = 0; aa < l_shm; aa++) {
            newval += Y[k*l_shm + aa]*M[aa]*vbark[aa*n_image + i];
        }
        newval = q2k[k*n_image + i] + sigma*newval;
        q2kp1[k*n_image + i] = newval;
        q2k[k*n_image + i] = newval;
    }
}

__global__ void DualKernel3(double *u1k, double *u2k, double *vk, double *wk,
                            double *u1bark, double *u2bark, double *vbark, double *wbark,
                            double *pk, double *gk, double *q0k, double *q1k, double *q2k,
                            double *pkp1, double *gkp1, double *q0kp1, double *q1kp1, double *q2kp1,
                            double *b, double *A, double *B, long *P,
                            double *f, double *Y, double *M, double sigma, double tau, double theta,
                            double lbd, double b_precond,
                            double *constraint_u, unsigned char *uconstrloc)
{
    /* This function makes heavy use of registers (34 32-bit registers), so
     * that it will not run with more than 960 threads per block on compute
     * capability 2.x!
     *
     * gkp1 = proj(gkp1, lbd)
     * gk = gkp1
     */

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
    if(i >= n_image || j >= m_gradients)
       return;

    // iteration variables and misc.
    int mm, tt;
    double *gij = &gkp1[i*msd_skip + j*sd_skip];
    double norm = 0.0;

#if (d_image == 1 || s_manifold == 1)
    for(mm = 0; mm < LIM; mm++) {
        norm += gij[mm*STEP1]*gij[mm*STEP1];
    }

    if(norm > lbd*lbd) {
        norm = lbd/sqrt(norm);
        for(mm = 0; mm < LIM; mm++) {
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
    for(mm = 0; mm < LIM; mm++) {
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

    if(smax > lbd) {
        // Compute orthonormal eigenvectors
        if(C12 == 0.0) {
            if(C11 >= C22) {
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
        for(mm = 0; mm < LIM; mm++) {
            // s1, s2 now used as temp. variables
            s1 = gij[mm*STEP1 + 0*STEP2];
            s2 = gij[mm*STEP1 + 1*STEP2];
            gij[mm*STEP1 + 0*STEP2] = s1*M11 + s2*M21;
            gij[mm*STEP1 + 1*STEP2] = s1*M12 + s2*M22;
        }
    }
#endif

    for(mm = 0; mm < s_manifold; mm++) {
        for(tt = 0; tt < d_image; tt++) {
            gk[i*msd_skip + j*sd_skip + mm*d_image + tt] = gij[mm*d_image + tt];
        }
    }
}
