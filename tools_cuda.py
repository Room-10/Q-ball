
from __future__ import division

import numpy as np

from pycuda import compiler, gpuarray
import pycuda.autoinit
from pycuda.driver import device_attribute as devattr

import logging

import util

def iterate_on_gpu(kernels, vars, max_iter):
    """ Execute a fixed number of iterations on the GPU

    Args:
        kernels, vars : Output of prepare_kernels(...)
        max_iter : Number of iterations to execute.
    Returns:
        The number of executed iterations.
    """
    gpu_itervars = [gpuarray.to_gpu(val) for name, val in vars['iter']]
    gpu_constvars = [val for name, val in vars['const']]

    with util.GracefulInterruptHandler() as interrupt_hdl:
        for _iter in range(max_iter):
            for kernel in kernels:
                kernel['func'].prepared_call(kernel['grid'], kernel['block'],
                    *gpu_constvars, *(v.gpudata for v in gpu_itervars))

            if interrupt_hdl.interrupted:
                logging.debug("GPU iteration interrupt (SIGINT) at iter=%d" % _iter)
                max_iter = _iter
                break

    [gv.get(ary=v[1]) for gv,v in zip(gpu_itervars, vars['iter'])]
    return max_iter

def prepare_kernels(files, templates, constvars, itervars):
    """ Compile and prepare CUDA kernel functions

    Args:
        files : list of cuda file names
        templates : list of tuples describing the kernels
        constvars : dict of readonly variables
        itervars : dict of readwrite variables
    Returns:
        A list of executable CUDA kernels for pd_iterate_on_gpu
    """
    vars = { 'const': list(constvars.items()), 'iter': list(itervars.items()) }
    preamble, signature = prepare_vars(vars)
    kernels_code = preamble
    for f in files:
        kernels_code += open(f, 'r').read()

    mod = compiler.SourceModule(kernels_code)
    kernels = [
        {
            'block': d[2],
            'grid': tuple(int(np.ceil(t/b)) for t,b in zip(d[1], d[2])),
            'func': mod.get_function(d[0])
        }
        for d in templates
    ]
    [k['func'].prepare(signature) for k in kernels]

    # print information on the current GPU
    attrs = pycuda.autoinit.device.get_attributes()
    blockdims = (attrs[devattr.MAX_BLOCK_DIM_X],
                 attrs[devattr.MAX_BLOCK_DIM_Y],
                 attrs[devattr.MAX_BLOCK_DIM_Z])
    griddims = (attrs[devattr.MAX_GRID_DIM_X],
                attrs[devattr.MAX_GRID_DIM_Y],
                attrs[devattr.MAX_GRID_DIM_Z])
    logging.info("CUDA kernels prepared for GPU (MAX_BLOCK_DIM={blockdims}, " \
        "MAX_GRID_DIM={griddims}, MAX_THREADS_PER_BLOCK={maxthreads})".format(
        blockdims="x".join(map(str,blockdims)),
        griddims="x".join(map(str,griddims)),
        maxthreads=attrs[devattr.MAX_THREADS_PER_BLOCK]
    ))

    return kernels, vars

def prepare_vars(vars):
    preamble = """
// hardcoded dimensions and respective skip sizes
"""
    signature = ""
    param_strings = []
    constvars = []
    for name, val in vars['const']:
        if type(val) is str:
            preamble += "#define %s ('%s')\n" % (name, val)
        elif type(val) is int or type(val) is np.int64:
            preamble += "#define %s (%d)\n" % (name, val)
        elif type(val) is float or type(val) is np.float64:
            signature += "d"
            param_strings.append("double %s" % name)
            constvars.append((name, val))
        elif type(val) is np.ndarray:
            if val.dtype == 'int64':
                preamble += "__constant__ int %s[%d] = { %s };\n" % (
                    name, val.size,
                    ", ".join(str(i) for i in val.ravel()))
            elif val.dtype == 'bool':
                preamble += "__constant__ unsigned char %s[%d] = { %s };\n" % (
                    name, val.size,
                    ", ".join(str(int(i)) for i in val.ravel()))
            elif val.dtype == 'float64':
                signature += "P"
                param_strings.append("double *%s" % name)
                constvars.append((name, gpuarray.to_gpu(val).gpudata))
            else:
                raise Exception("Numpy dtype not supported: %s" % val.dtype)
        else:
            raise Exception("Type not supported: %s" % type(val))

    for name, val in vars['iter']:
        if type(val) is np.ndarray:
            signature += "P"
            if val.dtype == 'float64':
                param_strings.append("double *%s" % name)
            else:
                raise Exception("Numpy dtype not supported: %s" % val.dtype)
        elif type(val) is float or type(val) is np.float64:
            signature += "d"
            param_strings.append("double %s" % name)
        else:
            raise Exception("Type not supported: %s" % type(val))

    preamble += "\n#define KERNEL_PARAMS %s\n" % ", ".join(param_strings)
    vars['const'] = constvars
    return preamble, signature