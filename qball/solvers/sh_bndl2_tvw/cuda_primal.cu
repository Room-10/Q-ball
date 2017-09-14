
__global__ void linop_adjoint1(double *xgrad, double *y)
{
    /* u1grad = 0
     * u1grad += diag(b) D' p (D' = -div with Dirichlet boundary)
     */

    SUBVAR_x_u1(u1grad,xgrad)
    SUBVAR_y_p(p,y)

    // global thread index
    int k = blockIdx.x*blockDim.x + threadIdx.x;

    // stay inside maximum dimensions
    if (k >= l_labels)
       return;

    // iteration variables and misc.
    int ii, aa, tt, is_boundary, curr_dim, curr_i, base;
    double fac = b[k]/(double)navgskips;

    // ugrad = 0
    for (ii = 0; ii < n_image; ii++) {
        u1grad[k*n_image + ii] = 0.0;
    }

    // u1grad += diag(b) D' p (D' = -div with Dirichlet boundary)
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
                    u1grad[k*n_image + (base + skips[tt])] +=
                        fac*p[k*nd_skip + tt*n_image + ii];
                    u1grad[k*n_image + base] -=
                        fac*p[k*nd_skip + tt*n_image + ii];
                }
            }
        }
    }
}

__global__ void linop_adjoint2(double *xgrad, double *y)
{
    /* wgrad^ij = A^j g_t^ij - B^j P^j p_t^i
     */

    SUBVAR_x_w(wgrad,xgrad)
    SUBVAR_y_p(p,y)
    SUBVAR_y_g(g,y)

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

    // wgrad[i,j,l,t]
    idx = i*msd_skip + j*sd_skip + l*d_image + t;
    newval = 0.0;

    // wgrad^ij = A^j g_t^ij
    for (mm = 0; mm < s_manifold; mm++) {
        // jlm,ijmt->ijlt
        newval += A[j*ss_skip + l*s_manifold + mm] *
                    g[i*msd_skip + j*sd_skip + mm*d_image + t];
    }
    // wgrad^ij -= B^j P^j p_t^i
    for (mm = 0; mm < r_points; mm++) {
        // jlm,jmti->ijlt
        newval -= B[j*sr_skip + l*r_points + mm] *
                    p[P[j*r_points + mm]*nd_skip + t*n_image + i];
    }
    wgrad[idx] = newval;
}

__global__ void linop_adjoint3(double *xgrad, double *y)
{
    /* u1grad += b q0' - q1
     * u2grad = -q2
     *
     * vgrad^i = Y'q1^i + M Y'q2^i
     */

    SUBVAR_x_u1(u1grad,xgrad)
    SUBVAR_x_u2(u2grad,xgrad)
    SUBVAR_x_v(vgrad,xgrad)
    SUBVAR_y_q0(q0,y)
    SUBVAR_y_q1(q1,y)
    SUBVAR_y_q2(q2,y)

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
        // u1grad[k,i]
        idx = k*n_image + i;
        newval = u1grad[idx];

        // u1grad += b q0' - q1
        newval += b[k]*(b_precond*q0[i]);
        newval -= q1[idx];
        u1grad[idx] = newval;

        if (inpaint_nloc[i]) {
            // u2grad = -q2
            u2grad[idx] = -q2[idx];
        }
    }

    if (k == 0) {
        // vgrad[m,i]
        idx = m*n_image + i;
        newval = 0.0;

        // vgrad^i = Y'q1^i
        for (kk = 0; kk < l_labels; kk++) {
            // km,ki->mi
            newval += Y[kk*l_shm + m]*q1[kk*n_image + i];
        }

        if (inpaint_nloc[i]) {
            // vgrad^i += M Y'q2^i
            for (kk = 0; kk < l_labels; kk++) {
                // km,ki->mi
                newval += Y[kk*l_shm + m]*M[m]*q2[kk*n_image + i];
            }
        }

        vgrad[idx] = newval;
    }
}

#ifdef precond
__global__ void prox_primal(double *x, double *xtau)
#else
__global__ void prox_primal(double *x, double tau)
#endif
{
    /* u1[~uconstrloc] = max(0, u1)
     * u1[uconstrloc] = constraint_u[uconstrloc]
     *
     * u2 = 1/(1 + tau*diag(b)) max(u2 + tau*diag(b)*fl,
     *                              min(u2 + tau*diag(b)*fu,
     *                                      (1 + tau*diag(b))*u2))
     */

    SUBVAR_x_u1(u1,x)
    SUBVAR_x_u2(u2,x)

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (i >= n_image || k >= l_labels)
       return;

    // u1[k,i]
    int idx = k*n_image + i;
    double newval;

    if (uconstrloc[i]) {
        // u[uconstrloc] = constraint_u[uconstrloc]
        newval = constraint_u[idx];
    } else {
        // ~uconstrloc
        newval = u1[idx];
        // u1 = max(0, u1)
        newval = fmax(0.0,  newval);
    }
    u1[idx] = newval;

    if (inpaint_nloc[i]) {
        // u2[k,i]
        idx = k*n_image + i;
        newval = u2[idx];
#ifdef precond
        SUBVAR_x_u2(u2tau,xtau)
        double tau = u2tau[idx];
#endif
        // u2 = 1/(1 + tau*diag(b)) max(u2 + tau*diag(b)*fl,
        //                              min(u2 + tau*diag(b)*fu,
        //                                      (1 + tau*diag(b))*u2))
        u2[idx] = 1.0/(1.0 + tau*b[k])*fmax(newval + tau*b[k]*fl[idx],
            fmin(newval + tau*b[k]*fu[idx], (1 + tau*b[k])*newval));
    }
}
