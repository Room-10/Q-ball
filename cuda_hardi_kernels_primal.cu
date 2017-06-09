
__global__ void PrimalKernel1(double *u1k, double *u2k, double *vk, double *wk,
                              double *u1bark, double *u2bark, double *vbark, double *wbark,
                              double *pk, double *gk, double *q0k, double *q1k, double *q2k,
                              double *pkp1, double *gkp1, double *q0kp1, double *q1kp1, double *q2kp1,
                              double *b, double *A, double *B, long *P,
                              double *f, double *Y, double *M, double sigma, double tau, double theta,
                              double lbd, double b_precond,
                              double *constraint_u, unsigned char *uconstrloc)
{
    /* u1bark = 0
     * u1bark += diag(b) D' pkp1 (D' = -div with Dirichlet boundary)
     */

    // global thread index
    int k = blockIdx.x*blockDim.x + threadIdx.x;

    // stay inside maximum dimensions
    if(k >= l_labels)
       return;

    // iteration variables and misc.
    int ii, aa, tt, is_boundary, curr_dim, curr_i, base;

    for(ii = 0; ii < n_image; ii++) {
        u1bark[k*n_image + ii] = 0.0;
    }

    for(ii = 0; ii < n_image; ii++) {
        // skip points on "bottom right" boundary
        is_boundary = false; curr_i = ii;
        for(tt = d_image - 1; tt >= 0; tt--) {
            curr_dim = curr_i / skips[tt];
            curr_i = curr_i % skips[tt];
            if(curr_dim == imagedims[d_image - 1 - tt] - 1) {
                is_boundary = true;
                break;
            }
        }

        if(!is_boundary) {
            for(tt = 0; tt < d_image; tt++) {
                for(aa = 0; aa < navgskips; aa++) {
                    base = ii + avgskips[tt*navgskips + aa];
                    u1bark[k*n_image + (base + skips[tt])] +=
                        b[k]/(double)navgskips * pkp1[k*nd_skip + tt*n_image + ii];
                    u1bark[k*n_image + base] -=
                        b[k]/(double)navgskips * pkp1[k*nd_skip + tt*n_image + ii];
                }
            }
        }
    }
}

__global__ void PrimalKernel2(double *u1k, double *u2k, double *vk, double *wk,
                              double *u1bark, double *u2bark, double *vbark, double *wbark,
                              double *pk, double *gk, double *q0k, double *q1k, double *q2k,
                              double *pkp1, double *gkp1, double *q0kp1, double *q1kp1, double *q2kp1,
                              double *b, double *A, double *B, long *P,
                              double *f, double *Y, double *M, double sigma, double tau, double theta,
                              double lbd, double b_precond,
                              double *constraint_u, unsigned char *uconstrloc)
{
    /* wbark^ij = A^j gkp1_t^ij - B^j P^j pkp1_t^i
     * wbark = wk - tau*wbark
     * wbark, wk = wbark + theta*(wbark - wk), wbark
     */

    // global thread index
    int _lj = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int t = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if(_lj >= sm_skip || i >= n_image || t >= d_image)
       return;

    // these have to be computed on the fly
    int l = _lj / m_gradients;
    int j = _lj % m_gradients;

    // iteration variable and misc.
    int mm;
    double wbark_tmp;

    wbark_tmp = 0.0;
    for(mm = 0; mm < s_manifold; mm++) {
        // jlm,ijmt->ijlt
        wbark_tmp += A[j*ss_skip + l*s_manifold + mm] *
                    gkp1[i*msd_skip + j*sd_skip + mm*d_image + t];
    }

    for(mm = 0; mm < r_points; mm++) {
        // jlm,jmti->ijlt
        wbark_tmp -= B[j*sr_skip + l*r_points + mm] *
                    pkp1[P[j*r_points + mm]*nd_skip + t*n_image + i];
    }

    wbark_tmp = wk[i*msd_skip + j*sd_skip + l*d_image + t] - tau*wbark_tmp;

    wbark[i*msd_skip + j*sd_skip + l*d_image + t] = wbark_tmp
        + theta*(wbark_tmp - wk[i*msd_skip + j*sd_skip + l*d_image + t]);
    wk[i*msd_skip + j*sd_skip + l*d_image + t] = wbark_tmp;
}

__global__ void PrimalKernel3(double *u1k, double *u2k, double *vk, double *wk,
                              double *u1bark, double *u2bark, double *vbark, double *wbark,
                              double *pk, double *gk, double *q0k, double *q1k, double *q2k,
                              double *pkp1, double *gkp1, double *q0kp1, double *q1kp1, double *q2kp1,
                              double *b, double *A, double *B, long *P,
                              double *f, double *Y, double *M, double sigma, double tau, double theta,
                              double lbd, double b_precond,
                              double *constraint_u, unsigned char *uconstrloc)
{
    /* u1bark += b q0kp1' - q1kp1
     * u1bark = u1k - tau*u1bark
     * u1bark[~uconstrloc] = max(0, u1bark)
     * u1bark[uconstrloc] = constraint_u[uconstrloc]
     * u1bark, u1k = u1bark + theta*(u1bark - u1k), u1bark
     *
     * u2bark = - q2kp1 - diag(b) f
     * u2bark = dataterm_factor*(u2k - tau*u2bark)
     * u2bark, u2k = u2bark + theta*(u2bark - u2k), u2bark
     */

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if(i >= n_image || k >= l_labels)
       return;

    // misc.
    double ubark_tmp;

    double dataterm_factor = 1.0/(1.0 + tau*b[k]);

    if(uconstrloc[i]) {
        ubark_tmp = constraint_u[k*n_image + i];
    } else {
        ubark_tmp = u1bark[k*n_image + i];
        ubark_tmp += b[k]*(b_precond*q0kp1[i]);
        ubark_tmp -= q1kp1[k*n_image + i];
        ubark_tmp = u1k[k*n_image + i] - tau*ubark_tmp;
        ubark_tmp = fmax(0.0,  ubark_tmp);
    }

    u1bark[k*n_image + i] = ubark_tmp + theta*(ubark_tmp - u1k[k*n_image + i]);
    u1k[k*n_image + i] = ubark_tmp;

    ubark_tmp = 0.0;
    ubark_tmp -= q2kp1[k*n_image + i];
    ubark_tmp -= b[k]*f[k*n_image + i];
    ubark_tmp = dataterm_factor*(u2k[k*n_image + i] - tau*ubark_tmp);

    u2bark[k*n_image + i] = ubark_tmp + theta*(ubark_tmp - u2k[k*n_image + i]);
    u2k[k*n_image + i] = ubark_tmp;
}

__global__ void PrimalKernel4(double *u1k, double *u2k, double *vk, double *wk,
                              double *u1bark, double *u2bark, double *vbark, double *wbark,
                              double *pk, double *gk, double *q0k, double *q1k, double *q2k,
                              double *pkp1, double *gkp1, double *q0kp1, double *q1kp1, double *q2kp1,
                              double *b, double *A, double *B, long *P,
                              double *f, double *Y, double *M, double sigma, double tau, double theta,
                              double lbd, double b_precond,
                              double *constraint_u, unsigned char *uconstrloc)
{
    /* vbark^i = Y'q1kp1^i + M Y'q2kp1^i
     * vbark = vk - tau*vbark
     * vbark, vk = vbark + theta*(vbark - vk), vbark
     */

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int m = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if(i >= n_image || m >= l_shm)
       return;

    // iteration variable and misc.
    int k;
    double vbark_tmp;

    vbark_tmp = 0.0;
    for(k = 0; k < l_labels; k++) {
        // km,ki->mi
        vbark_tmp += Y[k*l_shm + m]*(q1kp1[k*n_image + i] + M[m]*q2kp1[k*n_image + i]);
    }

    vbark_tmp = vk[m*n_image + i] - tau*vbark_tmp;

    vbark[m*n_image + i] = vbark_tmp + theta*(vbark_tmp - vk[m*n_image + i]);
    vk[m*n_image + i] = vbark_tmp;
}
