"""
Wasserstein distance
--------------------

Wasserstein-related functions.
"""

# NOTE: Wasserstein computation is very fast so parallelization is not necessary.

from ._wasserstein import _wdist_other, _wdist_self

__all__ = [
    "wdist",
]


def wdist(t, Qs1, Qs2):
    r"""Wasserstein distance matrix of 1D probability distributions.

    .. math::

        d_W(f_1, f_2)^2 = \int^1_0 (Q_1(t) - Q_2(t))^2 dt

    where :math:`Q_i` is the quantile function of :math:`f_i`.

    Parameters
    ----------
    t : (M,) ndarray
        Points over which *Qs1* and *Qs2* are measured.
        Must be strictly increasing from 0 to 1.
    Qs1 : (N1, M) ndarray
        Quantile functions of first set of probability distributions.
    Qs2 : (N2, M) ndarray or Non
        Quantile functions of second set of probability distributions.
        If ``None`` is passed, it is set to *Qs1*.

    Returns
    -------
    (N1, N2) array
        Wasserstein distance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import ProfileData
    >>> from heavyedge.wasserstein import quantile
    >>> from heavyedge_distance import get_sample_path
    >>> from heavyedge_distance.wasserstein import wdist
    >>> with ProfileData(get_sample_path("MeanProfiles-AreaScaled.h5")) as data:
    ...     x = data.x()
    ...     fs, Ls, _ = data[:]
    >>> t = np.linspace(0, 1, 100)
    >>> Qs = quantile(x, fs, Ls, t)
    >>> D1 = wdist(t, Qs, None)
    >>> D2 = wdist(t, Qs, Qs)
    """
    if Qs2 is None:
        return _wdist_self(t, Qs1)
    else:
        return _wdist_other(t, Qs1, Qs2)
