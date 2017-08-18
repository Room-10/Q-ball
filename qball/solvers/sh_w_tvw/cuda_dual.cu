
__global__ void DualKernel1(KERNEL_PARAMS)
{
    /* ggradkp1^ij = A^j' wkp1^ij
     * ggradbk = (1 + theta)*ggradkp1 - theta*ggradk
     * gkp1 = gk + sigma*ggradbk
     *
     * pgradkp1 = 0
     * pgradkp1_t^i += - P^j' B^j' wkp1_t^ij
     *
     * q0gradkp1 = b'ukp1
     * q0gradbk = (1 + theta)*q0gradkp1 - theta*q0gradk
     * q0kp1 = q0k + sigma*(q0gradbk - 1)
     *
     * (W1:)
     * p0gradkp1 = 0
     * p0gradkp1_t^i += - P^j' B^j' w0kp1_t^ij
     *
     * g0gradkp1^ij = A^j' w0kp1^ij
     * g0gradbk = (1 + theta)*g0gradkp1 - theta*g0gradk
     * g0kp1 = g0k + sigma*g0gradbk
     */

    SUBVAR_ukp1
    SUBVAR_wkp1
    SUBVAR_w0kp1
    SUBVAR_gk
    SUBVAR_gkp1
    SUBVAR_ggradbk
    SUBVAR_ggradk
    SUBVAR_ggradkp1
    SUBVAR_pgradkp1
    SUBVAR_q0k
    SUBVAR_q0kp1
    SUBVAR_q0gradbk
    SUBVAR_q0gradk
    SUBVAR_q0gradkp1
    SUBVAR_p0gradkp1
    SUBVAR_g0k
    SUBVAR_g0kp1
    SUBVAR_g0gradbk
    SUBVAR_g0gradk
    SUBVAR_g0gradkp1
#ifdef adaptive
    SUBVAR_sigma
#endif
#ifdef precond
    SUBVAR_gsigma
    SUBVAR_q0sigma
    SUBVAR_g0sigma
#endif

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

    // ggradkp1[i,j,m,t]
    idx = i*msd_skip + j*sd_skip + m*d_image + t;
    newval = 0.0;

    // ggradkp1^ij = A^j' wkp1^ij
    for (ll = 0; ll < s_manifold; ll++) {
        // jlm,ijlt->ijmt
        newval += A[j*ss_skip + ll*s_manifold + m] *
                    wkp1[i*msd_skip + j*sd_skip + ll*d_image + t];
    }
    ggradkp1[idx] = newval;

    // ggradbk = (1 + theta)*ggradkp1 - theta*ggradk
    ggradbk[idx] = (1 + theta)*newval - theta*ggradk[idx];

    // gkp1 = gk + sigma*ggradbk
#ifdef precond
    gkp1[idx] = gk[idx] + gsigma[idx]*ggradbk[idx];
#else
    gkp1[idx] = gk[idx] + sigma*ggradbk[idx];
#endif

    if (_mj == 0) {
        // pgradkp1 = 0
        for (kk = 0; kk < l_labels; kk++) {
            pgradkp1[kk*nd_skip + t*n_image + i] = 0.0;
        }

        // pgradkp1_t^i += - P^j' B^j' wkp1_t^ij
        for (jj = 0; jj < m_gradients; jj++) {
            for (mm = 0; mm < r_points; mm++) {
                idx = P[jj*r_points + mm]*nd_skip + t*n_image + i;
                newval = pgradkp1[idx];
                for (ll = 0; ll < s_manifold; ll++) {
                    // jlm,ijlt->jmti
                    newval -= B[jj*sr_skip + ll*r_points + mm] *
                                wkp1[i*msd_skip + jj*sd_skip + ll*d_image + t];
                }
                pgradkp1[idx] = newval;
            }
        }

        if (t == 0) {
            // q0gradkp1 = b'ukp1
            newval = 0.0;
            for (kk = 0; kk < l_labels; kk++) {
                newval += b[kk]*ukp1[kk*n_image + i];
            }
            newval *= b_precond;
            q0gradkp1[i] = newval;

            // q0gradbk = (1 + theta)*q0gradkp1 - theta*q0gradk
            q0gradbk[i] = (1 + theta)*newval - theta*q0gradk[i];

            // q0kp1 = q0k + sigma*(q0gradbk - 1)
#ifdef precond
            q0kp1[i] = q0k[i] + q0sigma[i]*(q0gradbk[i] - b_precond);
#else
            q0kp1[i] = q0k[i] + sigma*(q0gradbk[i] - b_precond);
#endif

#if 'W' == dataterm
            // p0gradkp1 = 0
            for (kk = 0; kk < l_labels; kk++) {
                p0gradkp1[kk*n_image + i] = 0.0;
            }

            // p0gradkp1_t^i += - P^j' B^j' w0kp1_t^ij
            for (jj = 0; jj < m_gradients; jj++) {
                for (mm = 0; mm < r_points; mm++) {
                    idx = P[jj*r_points + mm]*n_image + i;
                    newval = p0gradkp1[idx];
                    for (ll = 0; ll < s_manifold; ll++) {
                        // jlm,ijlt->jmti
                        newval -= B[jj*sr_skip + ll*r_points + mm] *
                                    w0kp1[i*sm_skip + jj*s_manifold + ll];
                    }
                    p0gradkp1[idx] = newval;
                }
            }
#endif
        }
    }

#if 'W' == dataterm
    if (t == 0) {
        // g0gradkp1[i,j,m]
        idx = i*sm_skip + j*s_manifold + m;
        newval = 0.0;

        // g0gradkp1^ij = A^j' w0kp1^ij
        for (ll = 0; ll < s_manifold; ll++) {
            // jlm,ijl->ijm
            newval += A[j*ss_skip + ll*s_manifold + m] *
                        w0kp1[i*sm_skip + j*s_manifold + ll];
        }
        g0gradkp1[idx] = newval;

        // g0gradbk = (1 + theta)*g0gradkp1 - theta*g0gradk
        g0gradbk[idx] = (1 + theta)*newval - theta*g0gradk[idx];

        // g0kp1 = g0k + sigma*g0gradbk
#ifdef precond
        g0kp1[idx] = g0k[idx] + g0sigma[idx]*g0gradbk[idx];
#else
        g0kp1[idx] = g0k[idx] + sigma*g0gradbk[idx];
#endif
    }
#endif
}

__global__ void DualKernel2(KERNEL_PARAMS)
{
    /* pgradkp1 += diag(b) D ukp1 (D is the gradient on a staggered grid)
     * pgradbk = (1 + theta)*pgradkp1 - theta*pgradk
     * pkp1 = pk + sigma*pgradbk
     *
     * q1gradkp1 = Y vkp1 - ukp1
     * q1gradbk = (1 + theta)*q1gradkp1 - theta*q1gradk
     * q1kp1 = q1k + sigma*q1gradbk
     *
     * (W1:)
     * p0gradkp1 += diag(b) ukp1
     * p0gradbk = (1 + theta)*p0gradkp1 - theta*p0gradk
     * p0kp1 = p0k + sigma*(p0gradbk - diag(b) f)
     */

    SUBVAR_ukp1
    SUBVAR_vkp1
    SUBVAR_pk
    SUBVAR_pkp1
    SUBVAR_pgradbk
    SUBVAR_pgradk
    SUBVAR_pgradkp1
    SUBVAR_q1k
    SUBVAR_q1kp1
    SUBVAR_q1gradbk
    SUBVAR_q1gradk
    SUBVAR_q1gradkp1
    SUBVAR_p0k
    SUBVAR_p0kp1
    SUBVAR_p0gradbk
    SUBVAR_p0gradk
    SUBVAR_p0gradkp1
#ifdef adaptive
    SUBVAR_sigma
#endif
#ifdef precond
    SUBVAR_psigma
    SUBVAR_q1sigma
    SUBVAR_p0sigma
#endif

    // global thread index
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int t = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if (k >= l_labels || i >= n_image || t >= d_image)
       return;

    // iteration variable and misc.
    int aa, base, idx;
    double newval;

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
    // pgradkp1[k,t,i]
    idx = k*nd_skip + t*n_image + i;
    newval = pgradkp1[idx];

    // pgradkp1 += diag(b) D ukp1 (D is the gradient on a staggered grid)
    if (!is_boundary) {
        for (aa = 0; aa < navgskips; aa++) {
            base = i + avgskips[t*navgskips + aa];
            newval += b[k]/(double)navgskips * (
                ukp1[k*n_image + (base + skips[t])] - ukp1[k*n_image + base]
            );
        }
    }
    pgradkp1[idx] = newval;

    // pgradbk = (1 + theta)*pgradkp1 - theta*pgradk
    pgradbk[idx] = (1 + theta)*newval - theta*pgradk[idx];

    // pkp1 = pk + sigma*pgradbk
#ifdef precond
    pkp1[idx] = pk[idx] + psigma[idx]*pgradbk[idx];
#else
    pkp1[idx] = pk[idx] + sigma*pgradbk[idx];
#endif

    if (t == 0) {
        // q1gradkp1[k,i]
        idx = k*n_image + i;
        newval = -ukp1[idx];

        // q1gradkp1 = Y vkp1 - ukp1
        for (aa = 0; aa < l_shm; aa++) {
            newval += Y[k*l_shm + aa]*vkp1[aa*n_image + i];
        }
        q1gradkp1[idx] = newval;

        // q1gradbk = (1 + theta)*q1gradkp1 - theta*q1gradk
        q1gradbk[idx] = (1 + theta)*newval - theta*q1gradk[idx];

        // q1kp1 = q1k + sigma*q1gradbk
#ifdef precond
        q1kp1[idx] = q1k[idx] + q1sigma[idx]*q1gradbk[idx];
#else
        q1kp1[idx] = q1k[idx] + sigma*q1gradbk[idx];
#endif

#if 'W' == dataterm
        // p0gradkp1[k,i]
        idx = k*n_image + i;
        newval = p0gradkp1[idx];

        // p0gradkp1 += diag(b) ukp1
        newval += b[k]*ukp1[idx];
        p0gradkp1[k*n_image + i] = newval;

        // p0gradbk = (1 + theta)*p0gradkp1 - theta*p0gradk
        p0gradbk[idx] = (1 + theta)*newval - theta*p0gradk[idx];

        // p0kp1 = p0k + sigma*(p0gradbk - diag(b) f)
        newval = p0gradbk[idx] - b[k]*f[idx];
#ifdef precond
        p0kp1[idx] = p0k[idx] + p0sigma[idx]*newval;
#else
        p0kp1[idx] = p0k[idx] + sigma*newval;
#endif
#endif
    }
}

__global__ void DualKernel3(KERNEL_PARAMS)
{
    /* This function makes heavy use of registers (34 32-bit registers), so
     * that it will not run with more than 960 threads per block on compute
     * capability 2.x!
     *
     * gkp1 = proj(gkp1, lbd)
     * g0kp1 = proj(g0kp1, 1.0) (W1)
     */

    SUBVAR_gkp1
    SUBVAR_g0kp1

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
    int mm, tt, idx;
    double *gij = &gkp1[i*msd_skip + j*sd_skip];
    double norm = 0.0;

    // gkp1 = proj(gkp1, lbd)
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
#if 'W' == dataterm
    gij = &g0kp1[i*sm_skip + j*s_manifold];
    norm = 0.0;

    // g0kp1 = proj(g0kp1, 1.0)
    for (mm = 0; mm < s_manifold; mm++) {
        norm += gij[mm]*gij[mm];
    }

    if (norm > 1.0) {
        norm = 1.0/sqrt(norm);
        for (mm = 0; mm < s_manifold; mm++) {
            gij[mm] *= norm;
        }
    }
#endif
}
