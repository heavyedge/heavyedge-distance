"""
Discrete Fréchet distance
-------------------------

1-D discrete Fréchet distance
"""

import numpy as np

from ._dfd import _dfd_1d_distmat

__all__ = [
    "dfd",
]


def dfd(Ys, Ls):
    """1-D discrete Fréchet distance matrix.

    Parameters
    ----------
    Ys : (N, M) ndarray
        Function curves.
    Ls : (N,) ndarray
        Length of supports of each *Ys*.

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
    ...     Ys, Ls, _ = data[:]
    >>> d = dfd(Ys, Ls)
    """
    return _dfd_1d_distmat(Ys, Ls.astype(np.int32))
