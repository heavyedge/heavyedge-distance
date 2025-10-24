"""
Discrete Fréchet distance
-------------------------

1-D discrete Fréchet distance
"""

import numpy as np

from ._dfd import _dfd_1d_distmat, _dfd_1d_distmat_self

__all__ = [
    "dfd",
]


def dfd(profiles1, profiles2, n_jobs):
    """1-D discrete Fréchet distance matrix.

    Parameters
    ----------
    profiles1 : tuple of (N1, M1) ndarray and (N1,) ndarray of int
        Function curves and their lengths of supports.
    profiles2 : tuple of (N2, M2) ndarray and (N2,) ndarray of int
        Function curves and their lengths of supports.
        If ``None`` is passed, it is set to *profiles1*.
    n_jobs : int
        Number of parallel workers.

    Returns
    -------
    (N1, N2) array
        Discrete Fréchet distance matrix.

    Notes
    -----
    ``dfd(profiles1, None)`` is faster than ``dfd(profiles1, profiles1)``.

    Examples
    --------
    >>> from heavyedge import ProfileData
    >>> from heavyedge_distance import get_sample_path
    >>> from heavyedge_distance.dfd import dfd
    >>> with ProfileData(get_sample_path("MeanProfiles-PlateauScaled.h5")) as data:
    ...     Zs, Ls, _ = data[:]
    >>> d1 = dfd((Zs, Ls), None, 1)
    >>> d2 = dfd((Zs, Ls), (Zs, Ls), 1)
    """
    Ys1, Ls1 = profiles1
    Ls1 = Ls1.astype(np.int32)
    if profiles2 is None:
        # dismat of profiles1 to itself (faster implementation)
        ret = _dfd_1d_distmat_self(Ys1, Ls1, n_jobs)
    else:
        Ys2, Ls2 = profiles2
        Ls2 = Ls2.astype(np.int32)
        ret = _dfd_1d_distmat(Ys1, Ls1, Ys2, Ls2, n_jobs)
    return ret
