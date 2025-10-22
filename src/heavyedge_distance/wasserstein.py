"""
Wasserstein distance
--------------------

Wasserstein-related functions.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ._wasserstein import optimize_q

__all__ = [
    "quantile",
    "wdist",
    "wmean",
]


def quantile(x, f, t):
    """Convert probability distributions to quantile functions.

    Parameters
    ----------
    x : (M,) ndarray
        Coordinates of grids over which *fs* are measured.
    fs : (N, M) ndarray
        Empirical probability density functions.
    t : (L,) ndarray
        Points over which the quantile function will be measured.
        Must be strictly increasing from 0 to 1.

    Returns
    -------
    (N, L) ndarray
        Quantile functions* over *t*.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_distance.api import scale_area
    >>> from heavyedge_distance.wasserstein import quantile
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, _, _ = data[:]
    ...     fs = scale_area(x, Ys)
    >>> t = np.linspace(0, 1, 100)
    >>> Qs = quantile(x, fs, t)
    """
    Gs = cumulative_trapezoid(f, x, initial=0, axis=-1)
    return _batch_interp(t, x, Gs, left=x[0], right=x[-1])


def _batch_interp(x, xp, fps, left, right):
    # np.interp for multiple arrays in vectorized way.
    N, M = fps.shape
    (L,) = x.shape

    # Find indices for interpolation positions
    idx = np.searchsorted(xp, x, side="left")
    idx = np.clip(idx, 1, M - 1)

    # Get xp and fp at neighboring indices
    x0 = xp[idx - 1]  # (L,)
    x1 = xp[idx]  # (L,)
    denom = x1 - x0

    # Interpolation weights (broadcasted over N)
    w = (x - x0) / denom  # (L,)
    w = w[None, :]  # shape (1, L) for broadcasting

    # Gather y0, y1 for all N at once
    y0 = fps[:, idx - 1]  # (N, L)
    y1 = fps[:, idx]  # (N, L)

    # Linear interpolation
    y = y0 + w * (y1 - y0)  # (N, L)

    # Handle out-of-bounds (like np.interp)
    left = np.full((N, 1), left)
    right = np.full((N, 1), right)

    # Apply left/right behavior
    y = np.where(x[None, :] < xp[0], left, y)
    y = np.where(x[None, :] > xp[-1], right, y)

    return y


def wdist(x1, f1, x2, f2, grid_num):
    r"""Wasserstein distance between two 1D probability distributions.

    .. math::

        d_W(f_1, f_2)^2 = \int^1_0 (Q_1(t) - Q_2(t))^2 dt

    where :math:`Q_i` is the quantile function of :math:`f_i`.

    Parameters
    ----------
    x1, f1 : ndarray
        The first empirical probability density function.
    x2, f2 : ndarray
        The second empirical probability density function.
    grid_num : int
        Number of sample points in [0, 1] to approximate the integral.

    Returns
    -------
    scalar
        Wasserstein distance.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_distance.wasserstein import wdist
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     (Y1, Y2), (L1, L2), _ = data[:2]
    >>> x1, Y1 = x[:L1], Y1[:L1]
    >>> x2, Y2 = x[:L2], Y2[:L2]
    >>> f1, f2 = Y1 / np.trapezoid(Y1, x1), Y2 / np.trapezoid(Y2, x2)
    >>> d = wdist(x1, f1, x2, f2, 100)
    """
    grid = np.linspace(0, 1, grid_num)
    Q1 = quantile(x1, f1.reshape(1, -1), grid)[0]
    Q2 = quantile(x2, f2.reshape(1, -1), grid)[0]
    return np.trapezoid((Q1 - Q2) ** 2, grid) ** 0.5


def wmean(xs, fs, grid_num):
    """Fréchet mean of probability distrubutions using Wasserstein metric.

    Parameters
    ----------
    xs : list of ndarray
        Points over which each distribution in *fs* is measured.
    fs : list of ndarray
        Empirical probability density functions.
    grid_num : int
        Number of sample points in [0, 1] to approximate the integral.

    Returns
    -------
    x, f : ndarray
        Probability density function.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_distance.wasserstein import wmean
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     (Y1, Y2), (L1, L2), _ = data[:2]
    >>> x1, Y1 = x[:L1], Y1[:L1]
    >>> x2, Y2 = x[:L2] + 3, Y2[:L2]
    >>> f1, f2 = Y1 / np.trapezoid(Y1, x1), Y2 / np.trapezoid(Y2, x2)
    >>> x, f = wmean([x1, x2], [f1, f2], 100)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(x1, f1, "--", color="gray")
    ... plt.plot(x2, f2, "--", color="gray")
    ... plt.plot(x, f)
    """
    grid = np.linspace(0, 1, grid_num)
    Q = np.array([quantile(x, f.reshape(1, -1), grid)[0] for x, f in zip(xs, fs)])
    g = np.mean(Q, axis=0)
    if np.all(np.diff(g) >= 0):
        q = g
    else:
        q = optimize_q(g)
    pdf = 1 / np.gradient(q, grid)
    pdf[-1] = 0
    pdf /= np.trapezoid(pdf, q)
    return q, pdf
