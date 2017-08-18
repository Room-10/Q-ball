
__global__ void PrimalKernel1(KERNEL_PARAMS)
{
    /* ukp1 = uk - tau*ugradk
     * ukp1 = 1/(1 + tau*diag(b))*(ukp1 + diag(b) f) (quadratic)
     * ukp1[~uconstrloc] = max(0, ukp1)
     * ukp1[uconstrloc] = constraint_u[uconstrloc]
     */

    SUBVAR_uk
    SUBVAR_ukp1
    SUBVAR_ugradk
#ifdef adaptive
    SUBVAR_tau
#endif
#ifdef precond
    SUBVAR_utau
#endif

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (i >= n_image || k >= l_labels)
       return;

    // ukp1[k,i]
    int idx = k*n_image + i;
    double newval;

    if (uconstrloc[i]) {
        // ukp1[uconstrloc] = constraint_u[uconstrloc]
        newval = constraint_u[idx];
    } else {
        // ~uconstrloc
        // ukp1 = uk - tau*ugradk
#ifdef precond
        newval = uk[idx] - utau[idx]*ugradk[idx];
#else
        newval = uk[idx] - tau*ugradk[idx];
#endif
#if 'Q' == dataterm
        // ukp1 = 1/(1 + tau*diag(b))*(ukp1 + diag(b) f)
#ifdef precond
        newval = 1.0/(1.0 + utau[idx]*b[k])*(newval + b[k]*f[idx]);
#else
        newval = 1.0/(1.0 + tau*b[k])*(newval + b[k]*f[idx]);
#endif
#endif
        // ukp1 = max(0, ukp1)
        newval = fmax(0.0,  newval);
    }
    ukp1[idx] = newval;
}

__global__ void PrimalKernel2(KERNEL_PARAMS)
{
    /* vkp1 = vk - tau*vgradk
     * wkp1 = wk - tau*wgradk
     * w0kp1 = w0k - tau*w0gradk (W1)
     */

    SUBVAR_vk
    SUBVAR_vkp1
    SUBVAR_vgradk
    SUBVAR_wk
    SUBVAR_wkp1
    SUBVAR_wgradk
    SUBVAR_w0k
    SUBVAR_w0kp1
    SUBVAR_w0gradk
#ifdef adaptive
    SUBVAR_tau
#endif
#ifdef precond
    SUBVAR_vtau
    SUBVAR_wtau
    SUBVAR_w0tau
#endif

    // global thread index
    int _lj = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int _mt = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if (_lj >= sm_skip || i >= n_image || _mt >= l_shm*d_image)
       return;

    // these have to be computed on the fly
    int l = _lj / m_gradients;
    int j = _lj % m_gradients;
    int m = _mt / d_image;
    int t = _mt % d_image;

    // misc.
    int idx;

    if (t == 0 && l == 0 && j == 0) {
        idx = m*n_image + i;
        // vkp1 = vk - tau*vgradk
#ifdef precond
        vkp1[idx] = vk[idx] - vtau[idx]*vgradk[idx];
#else
        vkp1[idx] = vk[idx] - tau*vgradk[idx];
#endif
    }

    if (m == 0) {
        idx = i*msd_skip + j*sd_skip + l*d_image + t;
        // wkp1 = wk - tau*wgradk
#ifdef precond
        wkp1[idx] = wk[idx] - wtau[idx]*wgradk[idx];
#else
        wkp1[idx] = wk[idx] - tau*wgradk[idx];
#endif
#if 'W' == dataterm
        if (t == 0) {
            idx = i*sm_skip + j*s_manifold + l;
            // w0kp1 = w0k - tau*w0gradk
#ifdef precond
            w0kp1[idx] = w0k[idx] - w0tau[idx]*w0gradk[idx];
#else
            w0kp1[idx] = w0k[idx] - tau*w0gradk[idx];
#endif
        }
#endif
    }
}

__global__ void PrimalKernel3(KERNEL_PARAMS)
{
    /* ugradkp1 = 0
     * ugradkp1 += diag(b) D' pkp1 (D' = -div with Dirichlet boundary)
     */

    SUBVAR_ugradkp1
    SUBVAR_pkp1

    // global thread index
    int k = blockIdx.x*blockDim.x + threadIdx.x;

    // stay inside maximum dimensions
    if (k >= l_labels)
       return;

    // iteration variables and misc.
    int ii, aa, tt, is_boundary, curr_dim, curr_i, base;

    // ugradkp1 = 0
    for (ii = 0; ii < n_image; ii++) {
        ugradkp1[k*n_image + ii] = 0.0;
    }

    // ugradkp1 += diag(b) D' pkp1 (D' = -div with Dirichlet boundary)
    for (ii = 0; ii < n_image; ii++) {
        // skip points on "bottom right" boundary
        is_boundary = false; curr_i = ii;
        for (tt = d_image - 1; tt >= 0; tt--) {
            curr_dim = curr_i / skips[tt];
            curr_i = curr_i % skips[tt];
            if (curr_dim == imagedims[d_image - 1 - tt] - 1) {
                is_boundary = true;
                break;
            }
        }

        if (!is_boundary) {
            for (tt = 0; tt < d_image; tt++) {
                for (aa = 0; aa < navgskips; aa++) {
                    base = ii + avgskips[tt*navgskips + aa];
                    ugradkp1[k*n_image + (base + skips[tt])] +=
                        b[k]/(double)navgskips * pkp1[k*nd_skip + tt*n_image + ii];
                    ugradkp1[k*n_image + base] -=
                        b[k]/(double)navgskips * pkp1[k*nd_skip + tt*n_image + ii];
                }
            }
        }
    }
}

__global__ void PrimalKernel4(KERNEL_PARAMS)
{
    /* wgradkp1^ij = A^j gkp1_t^ij - B^j P^j pkp1_t^i
     * w0gradkp1^ij = A^j g0kp1_t^ij - B^j P^j p0kp1_t^i (W1)
     */

    SUBVAR_wgradkp1
    SUBVAR_w0gradkp1
    SUBVAR_pkp1
    SUBVAR_gkp1
    SUBVAR_p0kp1
    SUBVAR_g0kp1

    // global thread index
    int _lj = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int t = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if (_lj >= sm_skip || i >= n_image || t >= d_image)
       return;

    // these have to be computed on the fly
    int l = _lj / m_gradients;
    int j = _lj % m_gradients;

    // iteration variable and misc.
    int mm, idx;
    double newval;

    // wgradkp1[i,j,l,t]
    idx = i*msd_skip + j*sd_skip + l*d_image + t;
    newval = 0.0;

    // wgradkp1^ij = A^j gkp1_t^ij
    for (mm = 0; mm < s_manifold; mm++) {
        // jlm,ijmt->ijlt
        newval += A[j*ss_skip + l*s_manifold + mm] *
                    gkp1[i*msd_skip + j*sd_skip + mm*d_image + t];
    }
    // wgradkp1^ij -= B^j P^j pkp1_t^i
    for (mm = 0; mm < r_points; mm++) {
        // jlm,jmti->ijlt
        newval -= B[j*sr_skip + l*r_points + mm] *
                    pkp1[P[j*r_points + mm]*nd_skip + t*n_image + i];
    }
    wgradkp1[idx] = newval;

#if 'W' == dataterm
    if (t == 0) {
        // w0gradkp1[i,j,l]
        idx = i*sm_skip + j*s_manifold + l;
        newval = 0.0;

        // w0gradkp1^ij = A^j g0kp1_t^ij
        for (mm = 0; mm < s_manifold; mm++) {
            // jlm,ijm->ijl
            newval += A[j*ss_skip + l*s_manifold + mm] *
                        g0kp1[i*sm_skip + j*s_manifold + mm];
        }
        // w0gradkp1^ij -= B^j P^j p0kp1_t^i
        for (mm = 0; mm < r_points; mm++) {
            // jlm,jmi->ijl
            newval -= B[j*sr_skip + l*r_points + mm] *
                        p0kp1[P[j*r_points + mm]*n_image + i];
        }
        w0gradkp1[idx] = newval;
    }
#endif
}

__global__ void PrimalKernel5(KERNEL_PARAMS)
{
    /* ugradkp1 += b q0kp1' - q1kp1
     * ugradkp1 += diag(b) p0kp1 (W1)
     *
     * vgradkp1^i = Y' q1kp1^i
     */

    SUBVAR_ugradkp1
    SUBVAR_vgradkp1
    SUBVAR_q0kp1
    SUBVAR_q1kp1
    SUBVAR_p0kp1

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;
    int m = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if (i >= n_image || k >= l_labels || m >= l_shm)
       return;

    // misc.
    int kk, idx;
    double newval;

    if (m == 0) {
        // ugradkp1[k,i]
        idx = k*n_image + i;
        newval = ugradkp1[idx];

        // ugradkp1 += b q0kp1' - q1kp1
        newval += b[k]*(b_precond*q0kp1[i]);
        newval -= q1kp1[idx];
#if 'W' == dataterm
        // ugradkp1 += diag(b) p0kp1
        newval += b[k]*p0kp1[idx];
#endif
        ugradkp1[idx] = newval;
    }

    if (k == 0) {
        // vgradkp1[m,i]
        idx = m*n_image + i;
        newval = 0.0;

        // vgradkp1^i = Y' q1kp1^i
        for (kk = 0; kk < l_labels; kk++) {
            // km,ki->mi
            newval += Y[kk*l_shm + m]*q1kp1[kk*n_image + i];
        }
        vgradkp1[idx] = newval;
    }
}

__global__ void PrimalKernel6(KERNEL_PARAMS)
{
#ifdef adaptive
    /* res_pk = |(xk - xkp1)/tauk - (xgradk - xgradkp1)|_1
     * res_dk = |(yk - ykp1)/sigmak - (ygradk - ygradkp1)|_1
     * if res_pk > s*res_dk*Delta ...
     */

    SUBVAR_tau
    SUBVAR_sigma

    // misc.
    int k;
    double newval;

    if (alphak[0] > 1e-10) {
        // res_pk = |(xk - xkp1)/tauk - (xgradk - xgradkp1)|_1
        newval = 0.0;
        for (k = 0; k < x_size; k++) {
            newval += fabs((xk[k] - xkp1[k])/tau - (xgradk[k] - xgradkp1[k]));
        }
        res_pk[0] = newval;

        //res_dk = |(yk - ykp1)/sigmak - (ygradk - ygradkp1)|_1
        newval = 0.0;
        for (k = 0; k < y_size; k++) {
            newval += fabs((yk[k] - ykp1[k])/sigma - (ygradk[k] - ygradkp1[k]));
        }
        res_dk[0] = newval;

        // if res_pk > s*res_dk*Delta ...
        if (res_pk[0] > s*res_dk[0]*Delta) {
            tauk[0] *= 1.0/(1.0 - alphak[0]);
            sigmak[0] *= (1.0 - alphak[0]);
            alphak[0] *= eta;
        }
        if (res_pk[0] < s*res_dk[0]/Delta) {
            tauk[0] *= (1.0 - alphak[0]);
            sigmak[0] *= 1.0/(1.0 - alphak[0]);
            alphak[0] *= eta;
        }
    }
#endif
}

__global__ void PrimalKernel7(KERNEL_PARAMS)
{
    /* xk, xgradk = xkp1, xgradkp1
     * yk, ygradk = ykp1, ygradkp1
     */

    long k = (blockIdx.y*blockDim.y + threadIdx.y)*65535
          + blockIdx.x*blockDim.x + threadIdx.x;

    if (k < x_size) {
        // xk, xgradk = xkp1, xgradkp1
        xk[k] = xkp1[k];
        xgradk[k] = xgradkp1[k];
    }

    if (k < y_size) {
        // yk, ygradk = ykp1, ygradkp1
        yk[k] = ykp1[k];
        ygradk[k] = ygradkp1[k];
    }
}
