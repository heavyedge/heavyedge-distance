"""Helper functions for wasserstein distance."""

cimport cython
cimport numpy as cnp
from libc.stdlib cimport free, malloc

import numpy as np

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=2] _wdist_self(double[:] t, double[:, :] Qs):
    cdef Py_ssize_t N = Qs.shape[0], M = t.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] ret = np.empty((N, N), dtype=np.float64)
    cdef Py_ssize_t i, j, k
    cdef double dist, dx

    # Main loops
    for i in range(N):
        ret[i, i] = 0.0
        for j in range(i + 1, N):
            dist = 0.0
            # Trapezoidal integration of squared quantile function differences
            for k in range(M - 1):
                dx = t[k + 1] - t[k]
                dist += 0.5 * dx * ((Qs[i, k] - Qs[j, k]) ** 2 + (Qs[i, k + 1] - Qs[j, k + 1]) ** 2)
            dist = dist ** 0.5
            ret[i, j] = dist
            ret[j, i] = dist

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=2] _wdist_other(double[:] t, double[:, :] Qs1, double[:, :] Qs2):
    cdef Py_ssize_t N1 = Qs1.shape[0], N2 = Qs2.shape[0], M = t.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] ret = np.empty((N1, N2), dtype=np.float64)
    cdef Py_ssize_t i, j, k
    cdef double dist, dx

    # Main loops
    for i in range(N1):
        for j in range(N2):
            dist = 0.0
            # Trapezoidal integration of squared quantile function differences
            for k in range(M - 1):
                dx = t[k + 1] - t[k]
                dist += 0.5 * dx * ((Qs1[i, k] - Qs2[j, k]) ** 2 + (Qs1[i, k + 1] - Qs2[j, k + 1]) ** 2)
            dist = dist ** 0.5
            ret[i, j] = dist

    return ret
