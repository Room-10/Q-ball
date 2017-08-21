
__global__ void linop_adjoint1(double *xgrad, double *y)
{
    /* ugrad = 0
     * ugrad += diag(b) D' p (D' = -div with Dirichlet boundary)
     */

    SUBVAR_x_u(ugrad,xgrad)
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
        ugrad[k*n_image + ii] = 0.0;
    }

    // ugrad += diag(b) D' p (D' = -div with Dirichlet boundary)
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
                    ugrad[k*n_image + (base + skips[tt])] +=
                        fac*p[k*nd_skip + tt*n_image + ii];
                    ugrad[k*n_image + base] -=
                        fac*p[k*nd_skip + tt*n_image + ii];
                }
            }
        }
    }
}

__global__ void linop_adjoint2(double *xgrad, double *y)
{
    /* wgrad^ij = A^j g_t^ij - B^j P^j p_t^i
     * w0grad^ij = A^j g0_t^ij - B^j P^j p0_t^i (W1)
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

#if 'W' == dataterm
    if (t == 0) {
        SUBVAR_x_w0(w0grad,xgrad)
        SUBVAR_y_p0(p0,y)
        SUBVAR_y_g0(g0,y)

        // w0grad[i,j,l]
        idx = i*sm_skip + j*s_manifold + l;
        newval = 0.0;

        // w0grad^ij = A^j g0_t^ij
        for (mm = 0; mm < s_manifold; mm++) {
            // jlm,ijm->ijl
            newval += A[j*ss_skip + l*s_manifold + mm] *
                        g0[i*sm_skip + j*s_manifold + mm];
        }
        // w0grad^ij -= B^j P^j p0_t^i
        for (mm = 0; mm < r_points; mm++) {
            // jlm,jmi->ijl
            newval -= B[j*sr_skip + l*r_points + mm] *
                        p0[P[j*r_points + mm]*n_image + i];
        }
        w0grad[idx] = newval;
    }
#endif
}

__global__ void linop_adjoint3(double *xgrad, double *y)
{
    /* ugrad += b q'
     * ugrad += diag(b) p0 (W1)
     */

    SUBVAR_x_u(ugrad,xgrad)
    SUBVAR_y_q(q,y)

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (i >= n_image || k >= l_labels)
       return;

    // misc.
    int idx;
    double newval;

    // ugrad[k,i]
    idx = k*n_image + i;
    newval = ugrad[idx];

    // ugrad += b q'
    newval += b[k]*(b_precond*q[i]);
#if 'W' == dataterm
    SUBVAR_y_p0(p0,y)
    // ugrad += diag(b) p0
    newval += b[k]*p0[idx];
#endif
    ugrad[idx] = newval;
}

#ifdef precond
__global__ void prox_primal(double *x, double *xtau)
#else
__global__ void prox_primal(double *x, double tau)
#endif
{
    /* u -= tau*diag(b) s (linear)
     * u = 1/(1 + tau*diag(b))*(u + tau*diag(b) f) (quadratic)
     * u[~uconstrloc] = max(0, u)
     * u[uconstrloc] = constraint_u[uconstrloc]
     */

    SUBVAR_x_u(u,x)

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (i >= n_image || k >= l_labels)
       return;

    // u[k,i]
    int idx = k*n_image + i;
    double newval;

    if (uconstrloc[i]) {
        // u[uconstrloc] = constraint_u[uconstrloc]
        newval = constraint_u[idx];
    } else {
        // ~uconstrloc
        newval = u[idx];
#if 'Q' == dataterm
#ifdef precond
        SUBVAR_x_u(utau,xtau)
        double tau = utau[idx];
#endif
        // u = 1/(1 + tau*diag(b))*(u + tau*diag(b) f)
        newval = 1.0/(1.0 + tau*b[k])*(newval + tau*b[k]*f[idx]);
#elif 'L' == dataterm
#ifdef precond
        SUBVAR_x_u(utau,xtau)
        double tau = utau[idx];
#endif
        // u -= tau*diag(b) s
        newval -= tau*b[k]*f[idx];
#endif
        // u = max(0, u)
        newval = fmax(0.0,  newval);
    }
    u[idx] = newval;
}
