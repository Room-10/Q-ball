
__global__ void DualKernel1(KERNEL_PARAMS)
{
    /* pkp1 = D vbark (D is the gradient on a staggered grid)
     * pkp1 = pk + sigma*pkp1
     */

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
    if(!is_boundary) {
        for(aa = 0; aa < navgskips; aa++) {
            base = i + avgskips[t*navgskips + aa];
            newval +=  fac * (
                vbark[k*n_image + (base + skips[t])] - vbark[k*n_image + base]
            );
        }
    }

    newval = pk[k*nd_skip + t*n_image + i] + sigma*newval;
    pkp1[k*nd_skip + t*n_image + i] = newval;
    pk[k*nd_skip + t*n_image + i] = newval;
}

__global__ void DualKernel2(KERNEL_PARAMS)
{
    /* q1kp1 = Y vbark - u1bark
     * q1kp1 = q1k + sigma*q1kp1
     * q1k = q1kp1
     *
     * q2kp1 = Y M vbark - u2bark
     * q2kp1 = q2k + sigma*q2kp1
     * q2k = q2kp1
     *
     * q0kp1 = b'u1bark - 1
     * q0kp1 = q0k + sigma*q0kp1
     * q0k = q0kp1
     *
     * pkp1 = proj(pkp1, lbd)
     * pk = pkp1
     */

    // global thread index
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if(k >= l_labels || i >= n_image)
       return;

    // iteration variables and misc.
    int mm, tt;
    double norm, newval;

    newval = -u1bark[k*n_image + i];
    for(mm = 0; mm < l_shm; mm++) {
        newval += Y[k*l_shm + mm]*vbark[mm*n_image + i];
    }
    newval = q1k[k*n_image + i] + sigma*newval;
    q1kp1[k*n_image + i] = newval;
    q1k[k*n_image + i] = newval;

    newval = -u2bark[k*n_image + i];
    for(mm = 0; mm < l_shm; mm++) {
        newval += Y[k*l_shm + mm]*M[mm]*vbark[mm*n_image + i];
    }
    newval = q2k[k*n_image + i] + sigma*newval;
    q2kp1[k*n_image + i] = newval;
    q2k[k*n_image + i] = newval;

    if(k == 0) {
        newval = 0.0;
        for(mm = 0; mm < l_labels; mm++) {
            newval += b[mm]*u1bark[mm*n_image + i];
        }
        newval = q0k[i] + sigma*b_precond*(newval - 1.0);
        q0kp1[i] = newval;
        q0k[i] = newval;

        norm = 0.0;
        for(mm = 0; mm < l_shm; mm++) {
            for(tt = 0; tt < d_image; tt++) {
                norm += pkp1[mm*nd_skip + tt*n_image + i] *
                                            pkp1[mm*nd_skip + tt*n_image + i];
            }
        }

        if(norm > lbd*lbd) {
            norm = lbd/sqrt(norm);
            for(mm = 0; mm < l_shm; mm++) {
                for(tt = 0; tt < d_image; tt++) {
                    pkp1[mm*nd_skip + tt*n_image + i] *= norm;
                }
            }
        }

        for(mm = 0; mm < l_shm; mm++) {
            for(tt = 0; tt < d_image; tt++) {
                pk[mm*nd_skip + tt*n_image + i] =
                                            pkp1[mm*nd_skip + tt*n_image + i];
            }
        }
    }
}
