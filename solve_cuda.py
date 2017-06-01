
from manifold_sphere import load_sphere
from tools import normalize_odf
from tools_diff import staggered_diff_avgskips
from solve_pd import pd_iteration_step, compute_primal_obj, compute_dual_obj
import util

import numpy as np

import logging

def l2_w1tv_fitting(data, gtab, sampling_matrix, model_matrix, lbd=1e+5):
    pass

def w1_tv_regularization(f, gtab,
        sampling_matrix=None,
        lbd=10.0,
        term_relgap=1e-7,
        term_infeas=None,
        term_maxiter=1000,
        step_bound=0.05,
        step_factor=0.005,
        granularity=100,
        use_gpu=True,
        constraint_u=None,
        dataterm="W1",
        continue_at=None
    ):
    """ Solve ...

    Args:
        f : reference image
        gtab : bvals and bvecs
        ... : more keyword arguments
    Returns:
        pd_state : the solution; a tuple of numpy arrays
                        (uk, vk, wk, w0k, pk, gk, q0k, q1k, p0k, g0k)
                   that can be put back into this function as the `continue_at`
                   parameter.
        details : dictionary containing information on the objective primal
                  and dual functions (including feasibility and pd-gap).
    """
    b_vecs = gtab.bvecs[gtab.bvals > 0,...].T
    b_sph = load_sphere(vecs=b_vecs)
    normalize_odf(f, b_sph.b)

    imagedims = f.shape[1:]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    l_labels = b_sph.mdims['l_labels']
    s_manifold = b_sph.mdims['s_manifold']
    m_gradients = b_sph.mdims['m_gradients']
    assert(f.shape[0] == l_labels)

    Y = np.eye(l_labels)
    if sampling_matrix is not None:
        Y = sampling_matrix
    l_shm = Y.shape[1]

    logging.info("Solving ({l_labels} labels, {l_shm} shm, m={m}; img: {imagedims}; " \
        "dataterm: {dataterm}, lambda={lbd:.3g}, steps<{maxiter})...".format(
        lbd=lbd,
        m=m_gradients,
        l_labels=l_labels,
        l_shm=l_shm,
        imagedims="x".join(map(str,imagedims)),
        dataterm=dataterm,
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
        uk = np.zeros((l_labels, n_image), order='C')
        uk[:] = np.tile(1.0/b_sph.b, (n_image, 1)).T/l_labels
        uk = uk.reshape((l_labels,) + imagedims)
        uk[:,uconstrloc] = constraint_u[:,uconstrloc]

        vk = np.zeros((l_shm, n_image), order='C')
        wk = np.zeros((n_image, m_gradients, s_manifold, d_image), order='C')
        w0k = np.zeros((n_image, m_gradients, s_manifold), order='C')
        pk = np.zeros((l_labels, d_image, n_image), order='C')
        gk = np.zeros((n_image, m_gradients, s_manifold, d_image), order='C')
        q0k = np.zeros(n_image)
        q1k = np.zeros((l_labels, n_image), order='C')
        p0k = np.zeros((l_labels, n_image), order='C')
        g0k = np.zeros((n_image, m_gradients, s_manifold), order='C')
    else:
        uk, vk, wk, w0k, pk, gk, q0k, q1k, p0k, g0k = (ar.copy() for ar in continue_at)

    ukp1 = uk.copy()
    vkp1 = vk.copy()
    wkp1 = wk.copy()
    w0kp1 = w0k.copy()
    pkp1 = pk.copy()
    gkp1 = gk.copy()
    q0kp1 = q0k.copy()
    q1kp1 = q1k.copy()
    p0kp1 = p0k.copy()
    g0kp1 = g0k.copy()
    ubark = uk.copy()
    vbark = vk.copy()
    wbark = wk.copy()
    w0bark = w0k.copy()
    g_norms = np.zeros((n_image, m_gradients), order='C')

    avgskips = staggered_diff_avgskips(imagedims)

    if dataterm == "quadratic":
        dataterm_factor = 1.0/(1.0 + tau*b_sph.b)
    elif dataterm == "W1":
        dataterm_factor = np.ones((l_labels,))
    else:
        raise Exception("Dataterm '%s' not supported!" % dataterm)

    if use_gpu:
        from cuda_kernels import prepare_const_gpudata, prepare_kernels
        from cuda_iterate import pd_iterate_on_gpu
        const_gpudata = prepare_const_gpudata(b_sph, f, Y, constraint_u, uconstrloc)
        prepared_kernels = prepare_kernels(uk, vk, wk, b_sph, avgskips, dataterm)

    with util.GracefulInterruptHandler() as interrupt_hdl:
        _iter = 0
        while _iter < term_maxiter:
            if use_gpu:
                iterations = pd_iterate_on_gpu(uk, vk, wk, w0k,
                                  ubark, vbark, wbark, w0bark,
                                  ukp1, vkp1, wkp1, w0kp1,
                                  pk, gk, q0k, q1k, p0k, g0k,
                                  pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1,
                                  sigma, tau, lbd, theta,
                                  const_gpudata, prepared_kernels,
                                  granularity)
                _iter += iterations
                if iterations < granularity:
                    interrupt_hdl.handle(None, None)
            else:
                pd_iteration_step(uk, vk, wk, w0k,
                                  ubark, vbark, wbark, w0bark,
                                  ukp1, vkp1, wkp1, w0kp1,
                                  pk, gk, q0k, q1k, p0k, g0k,
                                  pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1,
                                  sigma, tau, lbd, theta, dataterm_factor,
                                  b_sph, f, Y, constraint_u, uconstrloc,
                                  dataterm, avgskips, g_norms)
                _iter += 1

            if interrupt_hdl.interrupted or _iter % granularity == 0:
                obj_p, infeas_p = compute_primal_obj(ukp1, vkp1, wkp1, w0kp1,
                                        pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1,
                                        lbd, f, Y, b_sph, constraint_u, uconstrloc,
                                        dataterm, avgskips, g_norms)
                obj_d, infeas_d = compute_dual_obj(ukp1, vkp1, wkp1, w0kp1,
                                        pkp1, gkp1, q0kp1, q1kp1, p0kp1, g0kp1,
                                        lbd, f, Y, b_sph, constraint_u, uconstrloc,
                                        dataterm, avgskips, g_norms)

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

    return (uk, vk, wk, w0k, pk, gk, q0k, q1k, p0k, g0k), {
        'objp': obj_p,
        'objd': obj_d,
        'infeasp': infeas_p,
        'infeasd': infeas_d,
        'relgap': relgap
    }
