
__global__ void PrimalKernel1(KERNEL_PARAMS)
{
    /* vbark = 0
     * vbark += D' pkp1 (D' = -div with Dirichlet boundary)
     */

    SUBVAR_vbark
    SUBVAR_pkp1

    // global thread index
    int k = blockIdx.x*blockDim.x + threadIdx.x;

    // stay inside maximum dimensions
    if(k >= l_shm)
       return;

    // iteration variables and misc.
    int ii, aa, tt, is_boundary, curr_dim, curr_i, base;
    double fac = 1.0/(double)navgskips;

    for(ii = 0; ii < n_image; ii++) {
        vbark[k*n_image + ii] = 0.0;
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
                    vbark[k*n_image + (base + skips[tt])] +=
                        fac * pkp1[k*nd_skip + tt*n_image + ii];
                    vbark[k*n_image + base] -=
                        fac * pkp1[k*nd_skip + tt*n_image + ii];
                }
            }
        }
    }
}

__global__ void PrimalKernel2(KERNEL_PARAMS)
{
    /* u1bark = b q0kp1' - q1kp1
     * u1bark = u1k - tau*u1bark
     * u1bark[~uconstrloc] = max(0, u1bark)
     * u1bark[uconstrloc] = constraint_u[uconstrloc]
     * u1bark, u1k = u1bark + theta*(u1bark - u1k), u1bark
     *
     * u2bark = - q2kp1 - diag(b) f
     * u2bark = dataterm_factor*(u2k - tau*u2bark)
     * u2bark, u2k = u2bark + theta*(u2bark - u2k), u2bark
     */

    SUBVAR_u1k
    SUBVAR_u1bark
    SUBVAR_u1kp1
    SUBVAR_u2k
    SUBVAR_u2bark
    SUBVAR_u2kp1
    SUBVAR_q0kp1
    SUBVAR_q1kp1
    SUBVAR_q2kp1

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
        ubark_tmp = 0.0;
        ubark_tmp += b[k]*(b_precond*q0kp1[i]);
        ubark_tmp -= q1kp1[k*n_image + i];
        ubark_tmp = u1k[k*n_image + i] - tau*ubark_tmp;
        ubark_tmp = fmax(0.0,  ubark_tmp);
    }

    u1bark[k*n_image + i] = ubark_tmp + theta*(ubark_tmp - u1k[k*n_image + i]);
    u1k[k*n_image + i] = ubark_tmp;
    u1kp1[k*n_image + i] = ubark_tmp;

    ubark_tmp = 0.0;
    ubark_tmp -= q2kp1[k*n_image + i];
    ubark_tmp -= b[k]*f[k*n_image + i];
    ubark_tmp = dataterm_factor*(u2k[k*n_image + i] - tau*ubark_tmp);

    u2bark[k*n_image + i] = ubark_tmp + theta*(ubark_tmp - u2k[k*n_image + i]);
    u2k[k*n_image + i] = ubark_tmp;
    u2kp1[k*n_image + i] = ubark_tmp;
}

__global__ void PrimalKernel3(KERNEL_PARAMS)
{
    /* vbark^i += Y'q1kp1^i + M Y'q2kp1^i
     * vbark = vk - tau*vbark
     * vbark, vk = vbark + theta*(vbark - vk), vbark
     */

    SUBVAR_vk
    SUBVAR_vbark
    SUBVAR_vkp1
    SUBVAR_q1kp1
    SUBVAR_q2kp1

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int m = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if(i >= n_image || m >= l_shm)
       return;

    // iteration variable and misc.
    int k;
    double vbark_tmp;

    vbark_tmp = vbark[m*n_image + i];
    for(k = 0; k < l_labels; k++) {
        // km,ki->mi
        vbark_tmp += Y[k*l_shm + m]*(q1kp1[k*n_image + i] + M[m]*q2kp1[k*n_image + i]);
    }

    vbark_tmp = vk[m*n_image + i] - tau*vbark_tmp;

    vbark[m*n_image + i] = vbark_tmp + theta*(vbark_tmp - vk[m*n_image + i]);
    vk[m*n_image + i] = vbark_tmp;
    vkp1[m*n_image + i] = vbark_tmp;
}
