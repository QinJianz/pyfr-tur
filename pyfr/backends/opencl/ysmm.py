# -*- coding: utf-8 -*-

from ctypes import (POINTER, Structure, c_int, c_double, c_float, c_size_t,
                    c_uint, c_void_p, sizeof)

import numpy as np

from pyfr.backends.opencl.provider import OpenCLKernel
from pyfr.ctypesutil import LibWrapper


# Possible ysmm exception types
class YSMMError(Exception): pass


class YSMMSMM(Structure):
    _fields_ = [
        ('dtype', c_int),
        ('m', c_int),
        ('n', c_int),
        ('k', c_int),
        ('lda', c_int),
        ('ldb', c_int),
        ('ldc', c_int),
        ('alpha', c_double),
        ('beta', c_double),
        ('a', c_void_p),
        ('flags', c_int)
    ]


class YSMMWrappers(LibWrapper):
    _libname = 'ysmm'

    # Error codes
    _statuses = {
        '*': YSMMError
    }

    # Constants
    DTYPE_FP32 = 1
    DTYPE_FP64 = 2
    SUPPORT_LEVEL_TUNED = 3

    # Functions
    _functions = [
        (c_uint, 'libysmm_cl_get_support_level', c_void_p),
        (c_int, 'libysmm_cl_create_handle', POINTER(c_void_p), c_void_p,
         c_void_p, c_int),
        (None, 'libysmm_cl_destroy_handle', c_void_p),
        (c_int, 'libysmm_cl_create_smm_kernel', POINTER(c_void_p), c_void_p,
         POINTER(YSMMSMM), c_int, c_double),
        (None, 'libysmm_cl_destory_smm_kernel', c_void_p),
        (c_int, 'libysmm_cl_bind_smm_kernel', c_void_p, c_void_p, c_void_p),
        (c_int, 'libysmm_cl_enqueue_smm_kernel', c_void_p, c_void_p, c_uint,
         POINTER(c_void_p), POINTER(c_void_p)),
        (c_int, 'libysmm_cl_clone_smm_kernel', c_void_p, POINTER(c_void_p))
    ]


class OpenCLYSMMKernels:
    def __init__(self, backend):
        self.backend = backend
        self.handle = c_void_p()

        # Kernel cache
        self._kerns = {}

        # Load and wrap YSMM
        self.lib = lib = YSMMWrappers()

        # Query the support level for our device
        level = lib.libysmm_cl_get_support_level(backend.cl.dev)

        # If the level is reasonable then create a handle
        if level >= lib.SUPPORT_LEVEL_TUNED:
            lib.libysmm_cl_create_handle(self.handle, backend.cl.ctx,
                                         backend.cl.dev, 0)

            self.mul = self._mul

    def __del__(self):
        if self.handle:
            for k in self._kerns.values():
                self.lib.libysmm_cl_destory_smm_kernel(k)

            self.lib.libysmm_cl_destroy_handle(self.handle)

    def _mul(self, a, b, out, alpha=1.0, beta=0.0):
        cl = self.backend.cl
        w, h = self.lib, self.handle

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('YSMM requires a constant a matrix')

        # Dimensions
        ldb, ldc = b.leaddim, out.leaddim

        # Kernel handle
        kern = c_void_p()

        # Cache key
        ckey = (a.mid, alpha, beta, ldb, ldc)

        # Kernel is in the cache, clone it
        if ckey in self._kerns:
            ckern, dt = self._kerns[ckey]
            w.libysmm_cl_clone_smm_kernel(ckern, kern)
        else:
            # Fetch the matrix
            arr = np.ascontiguousarray(a.get())

            # Fill out the kernel structure
            smm = YSMMSMM()
            smm.dtype = w.DTYPE_FP64 if a.dtype == np.float64 else w.DTYPE_FP32
            smm.m = arr.shape[0]
            smm.n = b.ncol
            smm.k = arr.shape[1]
            smm.lda = arr.shape[1]
            smm.ldb = ldb
            smm.ldc = ldc
            smm.alpha = alpha
            smm.beta = beta
            smm.a = arr.ctypes.data

            # Create the kernel
            w.libysmm_cl_create_smm_kernel(kern, h, smm, sizeof(smm), 0)

            # Save a copy of the contents of the output matrix
            out_np = getattr(out, 'parent', out).get()

            def smmk(queue):
                evt_ptr = c_void_p()

                w.libysmm_cl_enqueue_smm_kernel(kern, queue, 0, None, evt_ptr)

                return cl.event(evt_ptr)

            # Benchmark the kernel
            dt = self._benchmark(smmk)

            # Store a clone of the kernel in the cache
            self._kerns[ckey] = (ckern := c_void_p()), dt
            w.libysmm_cl_clone_smm_kernel(kern, ckern)

            # Restore the output matrix
            getattr(out, 'parent', out).set(out_np)

        # Set the parameters
        w.libysmm_cl_bind_smm_kernel(kern, b, out)

        class MulKernel(OpenCLKernel):
            def __del__(self):
                w.libysmm_cl_destory_smm_kernel(kern)

            def run(self, queue, wait_for=None, ret_evt=False):
                if wait_for:
                    nevt = len(wait_for)
                    wait_for = (c_void_p*nevt)(*map(int, wait_for))
                else:
                    nevt = 0

                evt_ptr = c_void_p() if ret_evt else None

                w.libysmm_cl_enqueue_smm_kernel(kern, queue, nevt, wait_for,
                                                evt_ptr)

                if ret_evt:
                    return cl.event(evt_ptr)

        return MulKernel(mats=[b, out], dt=dt)
