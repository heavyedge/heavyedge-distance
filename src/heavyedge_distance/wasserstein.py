"""
Wasserstein distance
--------------------

Wasserstein-related functions.
"""

# NOTE: Wasserstein computation is very fast so parallelization is not necessary.

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ._wasserstein import _optimize_q, _quantile, _wdist_other, _wdist_self

__all__ = [
    "quantile",
    "wdist",
    "wmean",
]


def quantile(x, fs, Ls, t):
    """Convert probability distributions to quantile functions.

    Parameters
    ----------
    x : (M1,) ndarray
        Coordinates of grids over which *fs* are measured.
    fs : (N, M1) ndarray
        Empirical probability density functions.
    Ls : (N,) ndarray
        Length of supports of each *fs*.
    t : (M2,) ndarray
        Points over which the quantile function will be measured.
        Must be strictly increasing from 0 to 1.

    Returns
    -------
    (N, M2) ndarray
        Quantile functions* over *t*.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_distance.api import scale_area
    >>> from heavyedge_distance.wasserstein import quantile
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    ...     fs = scale_area(x, Ys)
    >>> t = np.linspace(0, 1, 100)
    >>> Qs = quantile(x, fs, Ls, t)
    """
    Gs = cumulative_trapezoid(fs, x, initial=0, axis=-1)
    return _quantile(x, Gs, Ls.astype(np.int32), t)


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
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_distance.api import scale_area
    >>> from heavyedge_distance.wasserstein import quantile, wdist
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    ...     fs = scale_area(x, Ys)
    >>> t = np.linspace(0, 1, 100)
    >>> Qs = quantile(x, fs, Ls, t)
    >>> D1 = wdist(t, Qs, None)
    >>> D2 = wdist(t, Qs, Qs)
    """
    if Qs2 is None:
        return _wdist_self(t, Qs1)
    else:
        return _wdist_other(t, Qs1, Qs2)


def wmean(x, fs, Ls, grid_num):
    """Fréchet mean of probability distrubutions using Wasserstein metric.

    Parameters
    ----------
    x : (M,) ndarray
        Coordinates of grids over which *fs* are measured.
    fs : (N, M) ndarray
        Empirical probability density functions.
    Ls : (N,) ndarray
        Length of supports of each *fs*.
    grid_num : int
        Number of sample points in [0, 1] to approximate the integral.

    Returns
    -------
    f_mean : ndarray
        Fréchet mean of *fs* over *x*.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_distance.api import scale_area
    >>> from heavyedge_distance.wasserstein import wmean
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    ...     fs = scale_area(x, Ys)
    >>> f_mean = wmean(x, fs, Ls, 100)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(x, fs.T, "--", color="gray")
    ... plt.plot(x, f_mean)
    """
    grid = np.linspace(0, 1, grid_num)
    Qs = quantile(x, fs, Ls, grid)
    g = np.mean(Qs, axis=0)
    if np.all(np.diff(g) >= 0):
        q = g
    else:
        q = _optimize_q(g)
    pdf = 1 / np.gradient(q, grid)
    pdf[-1] = 0
    pdf /= np.trapezoid(pdf, q)
    return np.interp(x, q, pdf, left=pdf[0], right=pdf[-1])
