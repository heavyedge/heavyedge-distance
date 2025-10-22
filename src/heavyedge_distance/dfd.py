"""
Discrete Fréchet distance
-------------------------

1-D discrete Fréchet distance
"""

import os

import numpy as np

from ._dfd import _dfd_1d_distmat

__all__ = [
    "dfd",
]


def dfd(Ys, Ls, n_jobs=None):
    """1-D discrete Fréchet distance matrix.

    Parameters
    ----------
    Ys : (N, M) ndarray
        Function curves.
    Ls : (N,) ndarray
        Length of supports of each *Ys*.
    n_jobs : int, optional
        Number of parallel workers.
        If not passed, `HEAVYEDGE_MAX_WORKERS` environment variable is used.
        If the environment variable is invalid, set to 1.

    Returns
    -------
    (N, N) array
        Discrete Fréchet distance matrix.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_distance.dfd import dfd
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:2]
    >>> d = dfd(Ys, Ls)
    """
    if n_jobs is not None:
        MAX_WORKERS = n_jobs
    else:
        MAX_WORKERS = os.environ.get("HEAVYEDGE_MAX_WORKERS")
        if MAX_WORKERS is not None:
            MAX_WORKERS = int(MAX_WORKERS)
        else:
            MAX_WORKERS = 1
    Ls = Ls.astype(np.int32)

    return _dfd_1d_distmat(Ys, Ls, Ys, Ls, MAX_WORKERS)
