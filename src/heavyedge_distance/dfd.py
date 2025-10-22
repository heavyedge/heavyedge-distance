"""
Discrete Fréchet distance
-------------------------

1-D discrete Fréchet distance
"""

import os

import numpy as np

from ._dfd import _dfd_1d_distmat, _dfd_1d_distmat_self

__all__ = [
    "dfd",
]


def dfd(profiles1, profiles2=None, n_jobs=None):
    """1-D discrete Fréchet distance matrix.

    Parameters
    ----------
    profiles1 : tuple of (N1, M1) ndarray and (N1,) ndarray of int
        Function curves and their lengths of supports.
    profiles2 : tuple of (N2, M2) ndarray and (N2,) ndarray of int, optional
        Function curves and their lengths of supports.
        If not passed, it is set to *profiles1* itself.
    n_jobs : int, optional
        Number of parallel workers.
        If not passed, `HEAVYEDGE_MAX_WORKERS` environment variable is used.
        If the environment variable is invalid, set to 1.

    Returns
    -------
    (N1, N2) array
        Discrete Fréchet distance matrix.

    Notes
    -----
    If you want to compute the distance matrix of *profiles1* to itself,
    not passing *profiles2* is faster than passing the same data twice.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_distance.dfd import dfd
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:3]
    >>> d1 = dfd((Ys, Ls))
    >>> d2 = dfd((Ys, Ls), (Ys, Ls))
    """
    if n_jobs is not None:
        MAX_WORKERS = n_jobs
    else:
        MAX_WORKERS = os.environ.get("HEAVYEDGE_MAX_WORKERS")
        if MAX_WORKERS is not None:
            MAX_WORKERS = int(MAX_WORKERS)
        else:
            MAX_WORKERS = 1

    Ys1, Ls1 = profiles1
    Ls1 = Ls1.astype(np.int32)
    if profiles2 is None:
        # dismat of profiles1 to itself
        ret = _dfd_1d_distmat_self(Ys1, Ls1, MAX_WORKERS)
    else:
        Ys2, Ls2 = profiles2
        Ls2 = Ls2.astype(np.int32)
        ret = _dfd_1d_distmat(Ys1, Ls1, Ys2, Ls2, MAX_WORKERS)
    return ret
