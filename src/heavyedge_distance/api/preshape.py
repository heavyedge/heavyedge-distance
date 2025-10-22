"""Functions to acquire pre-shapes of profiles."""

import numpy as np

__all__ = [
    "scale_area",
    "scale_plateau",
]


def scale_area(x, Ys):
    """Scale profiles by their area under curve.

    Parameters
    ----------
    x : (M,) array
        X coordinates of profile grid.
    Ys : (N, M) array
        Array of N profiles. Values after contact points must be zero.

    Returns
    -------
    Ys_scaled : (N, M) array
        Area-scaled profiles.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_distance.api import scale_area
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> Ys_scale = scale_area(x, Ys)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(x, Ys.T, color="gray", alpha=0.5)
    ... plt.plot(x, Ys_scale.T)
    """
    return Ys / np.trapezoid(Ys, x, axis=1)[:, np.newaxis]


def scale_plateau(Ys):
    """Scale profiles by plateau height

    Parameters
    ----------
    Ys : (N, M) array
        Array of N profiles. Values after contact points must be zero.

    Returns
    -------
    Ys_scaled : (N, M) array
        Area-scaled profiles.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_distance.api import scale_plateau
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> Ys_scale = scale_plateau(Ys)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(x, Ys.T, color="gray", alpha=0.5)
    ... plt.plot(x, Ys_scale.T)
    """
    return Ys / Ys[:, [0]]
