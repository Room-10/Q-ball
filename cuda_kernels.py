
from __future__ import division

import numpy as np

from pycuda import compiler, gpuarray
import pycuda.autoinit
from pycuda.driver import device_attribute as devattr

import logging

def prepare_const_gpudata(b_sph, f, Y, constraint_u, uconstrloc):
    """ Preload relevant constant data to the GPU

    Args:
        b_sph : manifold to solve problem on
        f : cost vector (s) or reference image
        Y : sampling matrix
        constraint_u : constraint vector
        uconstrloc : constraint locations
    Returns:
        A dictionary with references to the gpudata for pd_iterate_on_gpu.
    """
    return {
        'b_precond': b_sph.b_precond,
        'b': gpuarray.to_gpu(b_sph.b).gpudata,
        'A': gpuarray.to_gpu(b_sph.A).gpudata,
        'B': gpuarray.to_gpu(b_sph.B).gpudata,
        'P': gpuarray.to_gpu(b_sph.P).gpudata,
        'f': gpuarray.to_gpu(f).gpudata,
        'Y': gpuarray.to_gpu(Y).gpudata,
        'uconstrloc': gpuarray.to_gpu(uconstrloc).gpudata,
        'constraint_u': gpuarray.to_gpu(constraint_u).gpudata
    }

device_constants_template = """
// hardcoded dimensions and respective skip sizes
#define navgskips (%(navgskips)d)
#define n_image (%(n_image)d)
#define m_gradients (%(m_gradients)d)
#define s_manifold (%(s_manifold)d)
#define d_image (%(d_image)d)
#define l_labels (%(l_labels)d)
#define r_points (%(r_points)d)
#define l_shm (%(l_shm)d)

#define nd_skip (%(d_image)d*%(n_image)d)
#define ld_skip (%(d_image)d*%(l_labels)d)
#define sd_skip (%(s_manifold)d*%(d_image)d)
#define ss_skip (%(s_manifold)d*%(s_manifold)d)
#define sr_skip (%(s_manifold)d*%(r_points)d)
#define sm_skip (%(s_manifold)d*%(m_gradients)d)
#define msd_skip (%(m_gradients)d*%(s_manifold)d*%(d_image)d)
#define ndl_skip (%(n_image)d*%(d_image)d*%(l_labels)d)

#define dataterm '%(dataterm)s'

__constant__ int imagedims[%(d_image)d] = { %(imagedims)s };
__constant__ int skips[%(d_image)d] = { %(skips)s };
__constant__ int avgskips[%(d_image)d*%(navgskips)d] = { %(avgskips)s };
"""

def prepare_kernels(u, v, w, b_sph, avgskips, dataterm):
    """ Compile and prepare CUDA kernel functions

    Args:
        u, v, w : arrays in shape of objective primals
        b_sph : manifold to solve problem on
        avgskips : output of staggered_diff_avgskips(u.shape[1:])
    Returns:
        A list of executable CUDA kernels for pd_iterate_on_gpu
    """
    l_labels = u.shape[0]
    imagedims = u.shape[1:]
    l_shm = v.shape[0]
    n_image, m_gradients, s_manifold, d_image = w.shape
    r_points = b_sph.mdims['r_points']
    navgskips =  1 << (d_image - 1)

    skips = (1,)
    for t in range(1,d_image):
        skips += (skips[-1]*imagedims[d_image-t],)
    skips = np.array(skips, dtype=np.int64, order='C')

    kernels_code = device_constants_template % {
        'imagedims': ", ".join(str(i) for i in imagedims),
        'l_labels': l_labels,
        'n_image': n_image,
        'm_gradients': m_gradients,
        's_manifold': s_manifold,
        'd_image': d_image,
        'r_points': r_points,
        'l_shm': l_shm,
        'navgskips': navgskips,
        'avgskips': ", ".join(str(i) for i in avgskips.ravel()),
        'skips': ", ".join(str(i) for i in skips),
        'dataterm': dataterm[0].upper()
    }
    kernels_code += open('cuda_kernels_primal.cu', 'r').read()
    kernels_code += open('cuda_kernels_dual.cu', 'r').read()

    # print information on the current GPU
    attrs = pycuda.autoinit.device.get_attributes()
    blockdims = (attrs[devattr.MAX_BLOCK_DIM_X],
                 attrs[devattr.MAX_BLOCK_DIM_Y],
                 attrs[devattr.MAX_BLOCK_DIM_Z])
    griddims = (attrs[devattr.MAX_GRID_DIM_X],
                attrs[devattr.MAX_GRID_DIM_Y],
                attrs[devattr.MAX_GRID_DIM_Z])

    mod = compiler.SourceModule(kernels_code)

    result = [
        ("DualKernel1", (s_manifold*m_gradients, n_image, d_image), (16, 16, 1)),
        ("DualKernel2", (l_labels, n_image, d_image), (16, 16, 1)),
        ("DualKernel3", (n_image, m_gradients, 1), (16, 16, 1)),
        ("PrimalKernel1", (l_labels, 1, 1), (16, 1, 1)),
        ("PrimalKernel2", (s_manifold*m_gradients, n_image, d_image), (16, 16, 1)),
        ("PrimalKernel3", (n_image, l_labels, 1), (16, 16, 1)),
        ("PrimalKernel4", (n_image, l_shm, 1), (16, 16, 1)),
    ]

    result = [
        {
            'block': d[2],
            'grid': tuple(int(np.ceil(t/b)) for t,b in zip(d[1], d[2])),
            'func': mod.get_function(d[0])
        }
        for d in result
    ]
    [k['func'].prepare("P"*26 + "d"*5 + "P"*2) for k in result]
    logging.info("CUDA kernels prepared for GPU (MAX_BLOCK_DIM={blockdims}, " \
        "MAX_GRID_DIM={griddims}, MAX_THREADS_PER_BLOCK={maxthreads})".format(
        blockdims="x".join(map(str,blockdims)),
        griddims="x".join(map(str,griddims)),
        maxthreads=attrs[devattr.MAX_THREADS_PER_BLOCK]
    ))
    return result
