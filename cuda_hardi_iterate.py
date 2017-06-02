
import util

import logging

from pycuda import gpuarray

def pd_iterate_on_gpu(u1k, u2k, vk, wk,
                      u1bark, u2bark, vbark, wbark,
                      u1kp1, u2kp1, vkp1, wkp1,
                      pk, gk, q0k, q1k, q2k,
                      pkp1, gkp1, q0kp1, q1kp1, q2kp1,
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
    u1k_gpu = gpuarray.to_gpu(u1k.reshape(u1k.shape[0], -1))
    u2k_gpu = gpuarray.to_gpu(u2k)
    vk_gpu = gpuarray.to_gpu(vk)
    wk_gpu = gpuarray.to_gpu(wk)
    u1bark_gpu = gpuarray.to_gpu(u1bark.reshape(u1bark.shape[0], -1))
    u2bark_gpu = gpuarray.to_gpu(u2bark)
    vbark_gpu = gpuarray.to_gpu(vbark)
    wbark_gpu = gpuarray.to_gpu(wbark)
    pk_gpu = gpuarray.to_gpu(pk)
    gk_gpu = gpuarray.to_gpu(gk)
    q0k_gpu = gpuarray.to_gpu(q0k)
    q1k_gpu = gpuarray.to_gpu(q1k)
    q2k_gpu = gpuarray.to_gpu(q2k)
    pkp1_gpu = gpuarray.to_gpu(pkp1)
    gkp1_gpu = gpuarray.to_gpu(gkp1)
    q0kp1_gpu = gpuarray.to_gpu(q0kp1)
    q1kp1_gpu = gpuarray.to_gpu(q1kp1)
    q2kp1_gpu = gpuarray.to_gpu(q2kp1)

    with util.GracefulInterruptHandler() as interrupt_hdl:
        for _iter in range(max_iter):
            for kernel in prepared_kernels:
                kernel['func'].prepared_call(kernel['grid'], kernel['block'],
                    u1k_gpu.gpudata, u2k_gpu.gpudata, vk_gpu.gpudata, wk_gpu.gpudata,
                    u1bark_gpu.gpudata, u2bark_gpu.gpudata, vbark_gpu.gpudata, wbark_gpu.gpudata,
                    pk_gpu.gpudata, gk_gpu.gpudata, q0k_gpu.gpudata, q1k_gpu.gpudata, q2k_gpu.gpudata,
                    pkp1_gpu.gpudata, gkp1_gpu.gpudata, q0kp1_gpu.gpudata, q1kp1_gpu.gpudata, q2kp1_gpu.gpudata,
                    c_gpudata['b'], c_gpudata['A'], c_gpudata['B'], c_gpudata['P'],
                    c_gpudata['f'], c_gpudata['Y'], c_gpudata['M'], sigma, tau, theta,
                    lbd, c_gpudata['b_precond'],
                    c_gpudata['constraint_u'], c_gpudata['uconstrloc'])

            if interrupt_hdl.interrupted:
                logging.debug("GPU iteration interrupt (SIGINT) at iter=%d" % _iter)
                max_iter = _iter
                break

    u1k_gpu.get(ary=u1k)
    u2k_gpu.get(ary=u2k)
    vk_gpu.get(ary=vk)
    wk_gpu.get(ary=wk)
    u1bark_gpu.get(ary=u1bark)
    u2bark_gpu.get(ary=u2bark)
    vbark_gpu.get(ary=vbark)
    wbark_gpu.get(ary=wbark)
    pk_gpu.get(ary=pk)
    gk_gpu.get(ary=gk)
    q0k_gpu.get(ary=q0k)
    q1k_gpu.get(ary=q1k)
    q2k_gpu.get(ary=q2k)
    pkp1_gpu.get(ary=pkp1)
    gkp1_gpu.get(ary=gkp1)
    q0kp1_gpu.get(ary=q0kp1)
    q1kp1_gpu.get(ary=q1kp1)
    q2kp1_gpu.get(ary=q2kp1)

    u1kp1[:] = u1k
    u2kp1[:] = u2k
    vkp1[:] = vk
    wkp1[:] = wk

    return max_iter

