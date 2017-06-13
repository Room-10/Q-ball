
import util

import logging

from pycuda import gpuarray

def pd_iterate_on_gpu(uk, wk, w0k,
                      ubark, wbark, w0bark,
                      ukp1, wkp1, w0kp1,
                      pk, gk, qk, p0k, g0k,
                      pkp1, gkp1, qkp1, p0kp1, g0kp1,
                      sigma, tau, lbd, theta,
                      c_gpudata, prepared_kernels,
                      max_iter):
    """ Execute a fixed number of primal-dual-iterations on the GPU

    Args:
        ... : Almost all relevant local variables from solve_manifold
        c_gpudata : Output of prepare_const_gpudata(...)
        prepared_kernels : Output of prepare_kernels(...)
        max_iter : Number of iterations to execute.
    Returns:
        nothing, the result is stored in place!
    """
    uk_gpu = gpuarray.to_gpu(uk.reshape(uk.shape[0], -1))
    wk_gpu = gpuarray.to_gpu(wk)
    w0k_gpu = gpuarray.to_gpu(w0k)
    ubark_gpu = gpuarray.to_gpu(ubark.reshape(ubark.shape[0], -1))
    wbark_gpu = gpuarray.to_gpu(wbark)
    w0bark_gpu = gpuarray.to_gpu(w0bark)
    pk_gpu = gpuarray.to_gpu(pk)
    gk_gpu = gpuarray.to_gpu(gk)
    qk_gpu = gpuarray.to_gpu(qk)
    p0k_gpu = gpuarray.to_gpu(p0k)
    g0k_gpu = gpuarray.to_gpu(g0k)
    pkp1_gpu = gpuarray.to_gpu(pkp1)
    gkp1_gpu = gpuarray.to_gpu(gkp1)
    qkp1_gpu = gpuarray.to_gpu(qkp1)
    p0kp1_gpu = gpuarray.to_gpu(p0kp1)
    g0kp1_gpu = gpuarray.to_gpu(g0kp1)

    with util.GracefulInterruptHandler() as interrupt_hdl:
        for _iter in range(max_iter):
            for kernel in prepared_kernels:
                kernel['func'].prepared_call(kernel['grid'], kernel['block'],
                    uk_gpu.gpudata, wk_gpu.gpudata, w0k_gpu.gpudata,
                    ubark_gpu.gpudata, wbark_gpu.gpudata, w0bark_gpu.gpudata,
                    pk_gpu.gpudata, gk_gpu.gpudata, qk_gpu.gpudata, p0k_gpu.gpudata, g0k_gpu.gpudata,
                    pkp1_gpu.gpudata, gkp1_gpu.gpudata, qkp1_gpu.gpudata, p0kp1_gpu.gpudata, g0kp1_gpu.gpudata,
                    c_gpudata['b'], c_gpudata['A'], c_gpudata['B'], c_gpudata['P'],
                    c_gpudata['f'], sigma, tau, theta,
                    lbd, c_gpudata['b_precond'],
                    c_gpudata['constraint_u'], c_gpudata['uconstrloc'])

            if interrupt_hdl.interrupted:
                logging.debug("GPU iteration interrupt (SIGINT) at iter=%d" % _iter)
                max_iter = _iter
                break

    uk_gpu.get(ary=uk)
    wk_gpu.get(ary=wk)
    w0k_gpu.get(ary=w0k)
    ubark_gpu.get(ary=ubark)
    wbark_gpu.get(ary=wbark)
    w0bark_gpu.get(ary=w0bark)
    pk_gpu.get(ary=pk)
    gk_gpu.get(ary=gk)
    qk_gpu.get(ary=qk)
    p0k_gpu.get(ary=p0k)
    g0k_gpu.get(ary=g0k)
    pkp1_gpu.get(ary=pkp1)
    gkp1_gpu.get(ary=gkp1)
    qkp1_gpu.get(ary=qkp1)
    p0kp1_gpu.get(ary=p0kp1)
    g0kp1_gpu.get(ary=g0kp1)

    ukp1[:] = uk
    wkp1[:] = wk
    w0kp1[:] = w0k

    return max_iter

