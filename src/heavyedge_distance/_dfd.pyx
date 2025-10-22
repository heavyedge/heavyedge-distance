# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# cython: infer_types=True
"""Helper functions for discrete Fréchet distance."""

cimport cython
cimport numpy as cnp
from cython.parallel cimport prange
from libc.math cimport NAN, fabs, fmax, fmin
from libc.stdlib cimport free, malloc

import numpy as np

cnp.import_array()


cdef double _dfd_1d(double[:] P, double[:] Q) nogil:
    """1-D DFD between two curves"""
    cdef Py_ssize_t i, j
    cdef Py_ssize_t p = P.shape[0]
    cdef Py_ssize_t q = Q.shape[0]
    cdef double prev_left, prev_diag, prev_low, ret
    cdef double *ca = <double *> malloc(q * sizeof(double))
    if ca == NULL:
        # Can't raise inside nogil, so signal error with NaN
        return NAN

    # Initialization
    ca[0] = fabs(P[0] - Q[0])
    for j in range(1, q):
        ca[j] = fmax(ca[j - 1], fabs(P[0] - Q[j]))

    # Dynamic programming
    for i in range(1, p):
        prev_left = ca[0]
        ca[0] = fmax(prev_left, fabs(P[i] - Q[0]))
        for j in range(1, q):
            prev_diag = prev_left
            prev_low = ca[j - 1]
            prev_left = ca[j]
            ca[j] = fmax(fmin(fmin(prev_left, prev_diag), prev_low), fabs(P[i] - Q[j]))

    ret = ca[q - 1]
    free(ca)
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=2] _dfd_1d_distmat(double[:, :] Ys1, cnp.int32_t[:] Ls1, double[:, :] Ys2, cnp.int32_t[:] Ls2, int n_threads):
    """1-D DFD matrix between different profile vectors."""
    cdef Py_ssize_t N1 = Ys1.shape[0], N2 = Ys2.shape[0]
    cdef Py_ssize_t total = N1 * N2
    cdef Py_ssize_t idx, i, j

    cdef cnp.ndarray[cnp.float64_t, ndim=2] D = np.empty((N1, N2), dtype=np.float64)

    if n_threads > 2:
        for idx in prange(total, nogil=True, num_threads=n_threads, schedule='static'):
            i = idx // N2
            j = idx % N2

            D[i, j] = _dfd_1d(Ys1[i, :Ls1[i]], Ys2[j, :Ls2[j]])
    else:
        for i in range(N1):
            for j in range(N2):
                D[i, j] = _dfd_1d(Ys1[i, :Ls1[i]], Ys2[j, :Ls2[j]])

    return D
