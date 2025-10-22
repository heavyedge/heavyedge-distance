"""Helper functions for wasserstein distance."""

cimport cython
cimport numpy as cnp
from libc.stdlib cimport free, malloc

import numpy as np

cnp.import_array()

cdef void _interp(
        double[:] x,
        double[:] xp,
        double[:] fp,
        double[:] out,
        long last_idx,
    ):
    cdef Py_ssize_t i, j, n = xp.shape[0], nx = x.shape[0]
    cdef double xval, x0, x1, f0, f1, slope

    for i in range(nx):
        xval = x[i]
        if xval < xp[0]:
            out[i] = fp[0]
        elif xval > xp[n-1]:
            out[i] = fp[last_idx]
            continue
        else:
            # TODO: use binary search here for speedup
            j = 0
            while j < n - 1 and xp[j+1] < xval:
                j += 1

        x0 = xp[j]
        x1 = xp[j+1]
        f0 = fp[j]
        f1 = fp[j+1]
        slope = (f1 - f0) / (x1 - x0)
        out[i] = f0 + slope * (xval - x0)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=2] _quantile(double[:] t, double[:, :] Gs, double[:] x, long[:] last_idxs):
    cdef Py_ssize_t i, N = Gs.shape[0], L = t.shape[0]
    cdef double right
    cdef cnp.ndarray[cnp.float64_t, ndim=2] ret = np.empty((N, L), dtype=np.float64)
    for i in range(N):
        _interp(t, Gs[i, :], x, ret[i, :], last_idxs[i])
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=1] optimize_q(double[:] g):
    # M = number of probability-space grid, N = number of distance-space grid
    cdef Py_ssize_t i, j, idx
    cdef double[:] y_vals = np.unique(g)
    cdef Py_ssize_t M = g.shape[0], N = y_vals.shape[0]
    # Should memory error occurs, may need to make ca 1d and overwrite during loop.
    cdef double *ca = <double *> malloc(M * N * sizeof(double))
    if not ca:
        raise MemoryError()
    cdef int *predecessor = <int *> malloc((M - 1) * N * sizeof(int))
    if not predecessor:
        raise MemoryError()

    # Compute costs
    for i in range(M):  # TODO: parallize this i-loop
        for j in range(N):
            ca[i * N + j] = (g[i] - y_vals[j]) ** 2

    # Accumulate costs
    cdef Py_ssize_t prev_min_j
    cdef double prev_min
    for i in range(1, M):
        prev_min_j = 0
        prev_min = ca[(i - 1) * N + prev_min_j]
        for j in range(N):
            if ca[(i - 1) * N + j] < prev_min:
                prev_min_j = j
                prev_min = ca[(i - 1) * N + j]
            ca[i * N + j] += prev_min
            predecessor[(i - 1) * N + j] = prev_min_j

    # Backtrack
    cdef cnp.ndarray[cnp.float64_t, ndim=1] q = np.empty(M, dtype=np.float64)
    idx = 0
    # Last column
    for j in range(1, N):
        if ca[(M - 1) * N + j] < ca[(M - 1) * N + idx]:
            idx = j
    q[M - 1] = y_vals[idx]
    free(ca)
    # Other columns
    for i in range(1, M):
        idx = predecessor[(M - 1 - i) * N + idx]
        q[M - 1 - i] = y_vals[idx]

    free(predecessor)
    return q


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=2] _wdist(double[:] x, double[:, :] Qs):
    cdef Py_ssize_t N = Qs.shape[0], M = Qs.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] ret = np.empty((N, N), dtype=np.float64)
    cdef Py_ssize_t i, j, k
    cdef double dist, dx

    # Main loops
    for i in range(N):
        ret[i, i] = 0.0
        for j in range(i + 1, N):
            dist = 0.0
            # Trapezoidal integration
            for k in range(M - 1):
                dx = x[k + 1] - x[k]
                dist += 0.5 * dx * ((Qs[i, k] - Qs[j, k]) ** 2 + (Qs[i, k + 1] - Qs[j, k + 1]) ** 2)
            dist = dist ** 0.5
            ret[i, j] = dist
            ret[j, i] = dist

    return ret
