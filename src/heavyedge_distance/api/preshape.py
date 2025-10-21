"""Functions to acquire pre-shapes of profiles."""

import numpy as np

__all__ = [
    "scale_area",
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
    >>> from heavyedge.api import fill_after
    >>> from heavyedge_distance.api import scale_area
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    ...     fill_after(Ys, Ls, 0)
    >>> Ys = scale_area(x, Ys)
    """
    return Ys / np.trapezoid(Ys, x, axis=1)[:, np.newaxis]
