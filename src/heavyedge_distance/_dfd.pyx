"""Helper functions for discrete Fréchet distance."""

cimport cython
cimport numpy as cnp
from libc.stdlib cimport free, malloc

import numpy as np

cnp.import_array()


cdef double _dfd_1d(double[:] P, double[:] Q):
    cdef Py_ssize_t i, j
    cdef int p = P.shape[0], q = Q.shape[0]
    cdef double prev_left, prev_diag, prev_low
    cdef double *ca = <double *> malloc(q * sizeof(double))
    if not ca:
        raise MemoryError()

    ca[0] = abs(P[0] - Q[0])
    for j in range(1, q):
        ca[j] = max(ca[j - 1], abs(P[0] - Q[j]))

    for i in range(1, p):
        prev_left = ca[0]
        ca[0] = max(prev_left, abs(P[i] - Q[0]))
        for j in range(1, q):
            prev_diag = prev_left
            prev_low = ca[j - 1]
            prev_left = ca[j]
            ca[j] = max(min(prev_left, prev_diag, prev_low), abs(P[i] - Q[j]))

    ret = ca[q - 1]
    free(ca)
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=2] _dfd_1d_distmat(double[:, :] Ys, cnp.int32_t[:] Ls):
    cdef Py_ssize_t N = Ys.shape[0], M = Ys.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] ret = np.empty((N, N), dtype=np.float64)
    cdef Py_ssize_t i, j
    cdef double dist

    for i in range(N):
        ret[i, i] = 0.0
        for j in range(i + 1, N):
            dist = _dfd_1d(Ys[i, :Ls[i]], Ys[j, :Ls[j]])
            ret[i, j] = dist
            ret[j, i] = dist
    return ret
