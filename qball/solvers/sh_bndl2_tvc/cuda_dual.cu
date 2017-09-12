
__global__ void linop1(double *x, double *ygrad)
{
    /* pgrad = D v (D is the gradient on a staggered grid)
     */

    SUBVAR_x_v(v,x)
    SUBVAR_y_p(pgrad,ygrad)

    // global thread index
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int t = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if(k >= l_shm || i >= n_image || t >= d_image)
       return;

    // iteration variable and misc.
    int aa, base;
    double newval, fac;

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

    newval = 0.0;
    fac = 1.0/(double)navgskips;

    // pgrad = D v (D is the gradient on a staggered grid)
    if(!is_boundary) {
        for(aa = 0; aa < navgskips; aa++) {
            base = i + avgskips[t*navgskips + aa];
            newval +=  fac*(
                v[k*n_image + (base + skips[t])] - v[k*n_image + base]
            );
        }
    }
    pgrad[k*nd_skip + t*n_image + i] = newval;
}

__global__ void linop2(double *x, double *ygrad)
{
    /* q0grad = b'u1
     *
     * q1grad = Yv - u1
     *
     * q2grad = YMv - u2
     */

    SUBVAR_x_u1(u1,x)
    SUBVAR_x_u2(u2,x)
    SUBVAR_x_v(v,x)
    SUBVAR_y_q0(q0grad,ygrad)
    SUBVAR_y_q1(q1grad,ygrad)
    SUBVAR_y_q2(q2grad,ygrad)

    // global thread index
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if(k >= l_labels || i >= n_image)
       return;

    // iteration variables and misc.
    int mm, tt, idx;
    double newval;

    if(k == 0) {
        // q0grad = b'u1
        newval = 0.0;
        for(mm = 0; mm < l_labels; mm++) {
            newval += b[mm]*u1[mm*n_image + i];
        }
        newval *= b_precond;
        q0grad[i] = newval;
    }

    // q1grad[k,i]
    idx = k*n_image + i;
    newval = -u1[idx];

    // q1grad = Yv - u1
    for(mm = 0; mm < l_shm; mm++) {
        newval += Y[k*l_shm + mm]*v[mm*n_image + i];
    }
    q1grad[idx] = newval;

    if (inpaint_nloc[i]) {
        // q2grad[k,i]
        idx = k*n_image + i;
        newval = -u2[idx];

        // q2grad = YMv - u2
        for(mm = 0; mm < l_shm; mm++) {
            newval += Y[k*l_shm + mm]*M[mm]*v[mm*n_image + i];
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
    /* p = proj(p, lbd)
     * q0 -= sigma
     */

    SUBVAR_y_p(p,y)
    SUBVAR_y_q0(q0,y)

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if(i >= n_image || k >= l_labels)
       return;

    // iteration variables and misc.
    int mm, tt, idx;
    double norm = 0.0, newval;

    if (k == 0) {
        for(mm = 0; mm < l_shm; mm++) {
            for(tt = 0; tt < d_image; tt++) {
                idx = mm*nd_skip + tt*n_image + i;
                norm += p[idx]*p[idx];
            }
        }

        if(norm > lbd*lbd) {
            norm = lbd/sqrt(norm);
            for(mm = 0; mm < l_shm; mm++) {
                for(tt = 0; tt < d_image; tt++) {
                    p[mm*nd_skip + tt*n_image + i] *= norm;
                }
            }
        }

#ifdef precond
        SUBVAR_y_q0(q0sigma,ysigma)
        q0[i] -= q0sigma[i]*b_precond;
#else
        q0[i] -= sigma*b_precond;
#endif
    }
}
