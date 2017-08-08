
__global__ void PrimalKernel1(KERNEL_PARAMS)
{
    /* ubark = 0
     * ubark += diag(b) D' pkp1 (D' = -div with Dirichlet boundary)
     */

    SUBVAR_ubark
    SUBVAR_pkp1

    // global thread index
    int k = blockIdx.x*blockDim.x + threadIdx.x;

    // stay inside maximum dimensions
    if(k >= l_labels)
       return;

    // iteration variables and misc.
    int ii, aa, tt, is_boundary, curr_dim, curr_i, base;

    for(ii = 0; ii < n_image; ii++) {
        ubark[k*n_image + ii] = 0.0;
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
                    ubark[k*n_image + (base + skips[tt])] +=
                        b[k]/(double)navgskips * pkp1[k*nd_skip + tt*n_image + ii];
                    ubark[k*n_image + base] -=
                        b[k]/(double)navgskips * pkp1[k*nd_skip + tt*n_image + ii];
                }
            }
        }
    }
}

__global__ void PrimalKernel2(KERNEL_PARAMS)
{
    /* wbark^ij = A^j gkp1_t^ij - B^j P^j pkp1_t^i
     * wbark = wk - tau*wbark
     * wbark, wk = wbark + theta*(wbark - wk), wbark
     *
     * (W1:)
     * w0bark^ij = A^j g0kp1_t^ij - B^j P^j p0kp1_t^i
     * w0bark = w0k - tau*w0bark
     * w0bark, w0k = w0bark + theta*(w0bark - w0k), w0bark
     */

    SUBVAR_wk
    SUBVAR_wbark
    SUBVAR_wkp1
    SUBVAR_w0k
    SUBVAR_w0bark
    SUBVAR_w0kp1
    SUBVAR_pkp1
    SUBVAR_gkp1
    SUBVAR_p0kp1
    SUBVAR_g0kp1

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
    double wkp1_tmp;

    wkp1_tmp = 0.0;
    for(mm = 0; mm < s_manifold; mm++) {
        // jlm,ijmt->ijlt
        wkp1_tmp += A[j*ss_skip + l*s_manifold + mm] *
                    gkp1[i*msd_skip + j*sd_skip + mm*d_image + t];
    }

    for(mm = 0; mm < r_points; mm++) {
        // jlm,jmti->ijlt
        wkp1_tmp -= B[j*sr_skip + l*r_points + mm] *
                    pkp1[P[j*r_points + mm]*nd_skip + t*n_image + i];
    }

    wkp1_tmp = wk[i*msd_skip + j*sd_skip + l*d_image + t] - tau*wkp1_tmp;

    wbark[i*msd_skip + j*sd_skip + l*d_image + t] =
        (1 + theta)*wkp1_tmp - theta*wk[i*msd_skip + j*sd_skip + l*d_image + t];
    wk[i*msd_skip + j*sd_skip + l*d_image + t] = wkp1_tmp;
    wkp1[i*msd_skip + j*sd_skip + l*d_image + t] = wkp1_tmp;

#if 'W' == dataterm
    if(t == 0) {
        wkp1_tmp = 0.0;
        for(mm = 0; mm < s_manifold; mm++) {
            // jlm,ijm->ijl
            wkp1_tmp += A[j*ss_skip + l*s_manifold + mm] *
                        g0kp1[i*sm_skip + j*s_manifold + mm];
        }
        for(mm = 0; mm < r_points; mm++) {
            // jlm,jmi->ijl
            wkp1_tmp -= B[j*sr_skip + l*r_points + mm] *
                        p0kp1[P[j*r_points + mm]*n_image + i];
        }
        wkp1_tmp = w0k[i*sm_skip + j*s_manifold + l] - tau*wkp1_tmp;

        w0bark[i*sm_skip + j*s_manifold + l] =
            (1 + theta)*wkp1_tmp - theta*w0k[i*sm_skip + j*s_manifold + l];
        w0k[i*sm_skip + j*s_manifold + l] = wkp1_tmp;
        w0kp1[i*sm_skip + j*s_manifold + l] = wkp1_tmp;
    }
#endif
}

__global__ void PrimalKernel3(KERNEL_PARAMS)
{
    /* ubark += b q0kp1' - q1kp1 - diag(b) f (quadratic)
     * ubark += b q0kp1' - q1kp1 + diag(b) p0kp1 (W1)
     * ubark = dataterm_factor*(uk - tau*ubark)
     * ubark[~uconstrloc] = max(0, ubark)
     * ubark[uconstrloc] = constraint_u[uconstrloc]
     * ubark, uk = ubark + theta*(ubark - uk), ubark
     */

    SUBVAR_uk
    SUBVAR_ubark
    SUBVAR_ukp1
    SUBVAR_q0kp1
    SUBVAR_q1kp1
    SUBVAR_p0kp1

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if(i >= n_image || k >= l_labels)
       return;

    // misc.
    double ukp1_tmp;

    double dataterm_factor = 1.0;
#if 'Q' == dataterm
    dataterm_factor = 1.0/(1.0 + tau*b[k]);
#endif

    if(uconstrloc[i]) {
        ukp1_tmp = constraint_u[k*n_image + i];
    } else {
        ukp1_tmp = ubark[k*n_image + i];
        ukp1_tmp += b[k]*(b_precond*q0kp1[i]);
        ukp1_tmp -= q1kp1[k*n_image + i];
#if 'Q' == dataterm
        ukp1_tmp -= b[k]*f[k*n_image + i];
#elif 'W' == dataterm
        ukp1_tmp += b[k]*p0kp1[k*n_image + i];
#endif
        ukp1_tmp = dataterm_factor*(uk[k*n_image + i] - tau*ukp1_tmp);
        ukp1_tmp = fmax(0.0,  ukp1_tmp);
    }

    ubark[k*n_image + i] = (1 + theta)*ukp1_tmp - theta*uk[k*n_image + i];
    uk[k*n_image + i] = ukp1_tmp;
    ukp1[k*n_image + i] = ukp1_tmp;
}

__global__ void PrimalKernel4(KERNEL_PARAMS)
{
    /* vbark^i = Y' q1kp1^i
     * vbark = vk - tau*vbark
     * vbark, vk = vbark + theta*(vbark - vk), vbark
     */

    SUBVAR_vk
    SUBVAR_vbark
    SUBVAR_vkp1
    SUBVAR_q1kp1

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int m = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if(i >= n_image || m >= l_shm)
       return;

    // iteration variable and misc.
    int k;
    double vkp1_tmp;

    vkp1_tmp = 0.0;
    for(k = 0; k < l_labels; k++) {
        // km,ki->mi
        vkp1_tmp += Y[k*l_shm + m]*q1kp1[k*n_image + i];
    }

    vkp1_tmp = vk[m*n_image + i] - tau*vkp1_tmp;

    vbark[m*n_image + i] = (1 + theta)*vkp1_tmp - theta*vk[m*n_image + i];
    vk[m*n_image + i] = vkp1_tmp;
    vkp1[m*n_image + i] = vkp1_tmp;
}
