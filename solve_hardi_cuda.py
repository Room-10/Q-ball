
from manifold_sphere import load_sphere
from tools_diff import staggered_diff_avgskips
from solve_hardi_pd import pd_iteration_step, compute_primal_obj, compute_dual_obj
import util

import numpy as np

import logging

def l2_w1tv_fitting(data, gtab, sampling_matrix, model_matrix,
        lbd=1.0,
        term_relgap=1e-7,
        term_infeas=None,
        term_maxiter=150000,
        step_bound=0.0012,
        step_factor=0.001,
        granularity=5000,
        use_gpu=True,
        constraint_u=None,
        continue_at=None
    ):
    """ Solve ...

    Args:
        data : reference HARDI image
        gtab : bvals and bvecs
        ... : more keyword arguments
    Returns:
        pd_state : the solution; a tuple of numpy arrays
                        (u1k, u2k, vk, wk, pk, gk, q0k, q1k, q2k)
                   that can be put back into this function as the `continue_at`
                   parameter.
        details : dictionary containing information on the objective primal
                  and dual functions (including feasibility and pd-gap).
    """
    b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
    b_sph = load_sphere(vecs=b_vecs)

    imagedims = data.shape[:-1]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    l_labels = b_sph.mdims['l_labels']
    s_manifold = b_sph.mdims['s_manifold']
    m_gradients = b_sph.mdims['m_gradients']
    assert(data.shape[-1] == l_labels)

    Y = np.zeros(sampling_matrix.shape, order='C')
    Y[:] = sampling_matrix
    l_shm = Y.shape[1]
    M = model_matrix
    assert(model_matrix.size == l_shm)

    f = np.zeros((l_labels, n_image), order='C')
    f[:] = np.log(-np.log(data)).reshape(-1, l_labels).T
    f_mean = np.einsum('ki,k->i', f, b_sph.b)/(4*np.pi)
    f -= f_mean

    logging.info("Solving ({l_labels} labels, {l_shm} shm, m={m}; " \
        "img: {imagedims}; lambda={lbd:.3g}, steps<{maxiter})...".format(
        lbd=lbd,
        m=m_gradients,
        l_labels=l_labels,
        l_shm=l_shm,
        imagedims="x".join(map(str,imagedims)),
        maxiter=term_maxiter
    ))

    if term_infeas is None:
        term_infeas = term_relgap

    if constraint_u is None:
        constraint_u = np.zeros((l_labels,) + imagedims, order='C')
        constraint_u[:] = np.nan
    uconstrloc = np.any(np.logical_not(np.isnan(constraint_u)), axis=0)

    bnd = step_bound   # < 1/|K|^2
    fact = step_factor # tau/sigma
    sigma = np.sqrt(bnd/fact)
    tau = bnd/sigma
    theta = 0.99

    obj_p = obj_d = infeas_p = infeas_d = relgap = 0.

    if continue_at is None:
        # start with a uniform distribution in each voxel
        u1k = np.ones((l_labels,) + imagedims, order='C')/np.einsum('k->', b_sph.b)
        u1k[:,uconstrloc] = constraint_u[:,uconstrloc]

        u2k = np.zeros((l_labels, n_image), order='C')
        vk = np.zeros((l_shm, n_image), order='C')
        vk[0,:] = .5 / np.sqrt(np.pi)
        wk = np.zeros((n_image, m_gradients, s_manifold, d_image), order='C')
        pk = np.zeros((l_labels, d_image, n_image), order='C')
        gk = np.zeros((n_image, m_gradients, s_manifold, d_image), order='C')
        q0k = np.zeros(n_image)
        q1k = np.zeros((l_labels, n_image), order='C')
        q2k = np.zeros((l_labels, n_image), order='C')
    else:
        u1k, u2k, vk, wk, pk, gk, q0k, q1k, q2k = (ar.copy() for ar in continue_at)

    u1kp1 = u1k.copy()
    u2kp1 = u2k.copy()
    vkp1 = vk.copy()
    wkp1 = wk.copy()
    pkp1 = pk.copy()
    gkp1 = gk.copy()
    q0kp1 = q0k.copy()
    q1kp1 = q1k.copy()
    q2kp1 = q2k.copy()
    u1bark = u1k.copy()
    u2bark = u2k.copy()
    vbark = vk.copy()
    wbark = wk.copy()
    g_norms = np.zeros((n_image, m_gradients), order='C')

    avgskips = staggered_diff_avgskips(imagedims)
    dataterm_factor = 1.0/(1.0 + tau*b_sph.b)

    if use_gpu:
        from cuda_hardi_kernels import prepare_const_gpudata, prepare_kernels
        from cuda_hardi_iterate import pd_iterate_on_gpu
        const_gpudata = prepare_const_gpudata(b_sph, f, Y, M, constraint_u, uconstrloc)
        prepared_kernels = prepare_kernels(u1k, u2k, vk, wk, b_sph, avgskips)

    with util.GracefulInterruptHandler() as interrupt_hdl:
        _iter = 0
        while _iter < term_maxiter:
            if use_gpu:
                iterations = pd_iterate_on_gpu(u1k, u2k, vk, wk,
                                  u1bark, u2bark, vbark, wbark,
                                  u1kp1, u2kp1, vkp1, wkp1,
                                  pk, gk, q0k, q1k, q2k,
                                  pkp1, gkp1, q0kp1, q1kp1, q2kp1,
                                  sigma, tau, lbd, theta,
                                  const_gpudata, prepared_kernels,
                                  granularity)
                _iter += iterations
                if iterations < granularity:
                    interrupt_hdl.handle(None, None)
            else:
                pd_iteration_step(u1k, u2k, vk, wk,
                                  u1bark, u2bark, vbark, wbark,
                                  u1kp1, u2kp1, vkp1, wkp1,
                                  pk, gk, q0k, q1k, q2k,
                                  pkp1, gkp1, q0kp1, q1kp1, q2kp1,
                                  sigma, tau, lbd, theta, dataterm_factor,
                                  b_sph, f, Y, M, constraint_u, uconstrloc,
                                  avgskips, g_norms)
                _iter += 1

            if interrupt_hdl.interrupted or _iter % granularity == 0:
                obj_p, infeas_p = compute_primal_obj(u1kp1, u2kp1, vkp1, wkp1,
                                        pkp1, gkp1, q0kp1, q1kp1, q2kp1,
                                        lbd, f, Y, M, b_sph, constraint_u, uconstrloc,
                                        avgskips, g_norms)
                obj_d, infeas_d = compute_dual_obj(u1kp1, u2kp1, vkp1, wkp1,
                                        pkp1, gkp1, q0kp1, q1kp1, q2kp1,
                                        lbd, f, Y, M, b_sph, constraint_u, uconstrloc,
                                        avgskips, g_norms)

                # compute relative primal-dual gap
                relgap = (obj_p - obj_d) / max(np.spacing(1), obj_d)

                logging.debug("#{:6d}: objp = {: 9.6g} ({: 9.6g}), " \
                    "objd = {: 9.6g} ({: 9.6g}), " \
                    "gap = {: 9.6g}, " \
                    "relgap = {: 9.6g} ".format(
                    _iter, obj_p, infeas_p,
                    obj_d, infeas_d,
                    obj_p - obj_d,
                    relgap
                ))

                if relgap < term_relgap and max(infeas_p, infeas_d) < term_infeas:
                    break

                if interrupt_hdl.interrupted:
                    break

    return (u1k, u2k, vk, wk, pk, gk, q0k, q1k, q2k), {
        'objp': obj_p,
        'objd': obj_d,
        'infeasp': infeas_p,
        'infeasd': infeas_d,
        'relgap': relgap
    }
