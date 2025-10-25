import os

import numpy as np
from heavyedge.wasserstein import quantile

from heavyedge_distance.dfd import dfd
from heavyedge_distance.wasserstein import wdist

__all__ = [
    "distmat_euclidean",
    "distmat_wasserstein",
    "distmat_frechet",
]


def _distmat(
    converter,
    distfunc,
    f1,
    f2=None,
    batch_size=None,
    n_jobs=None,
    logger=lambda x: None,
):
    if n_jobs is not None:
        pass
    else:
        n_jobs = os.environ.get("HEAVYEDGE_MAX_WORKERS")
        if n_jobs is not None:
            n_jobs = int(n_jobs)
        else:
            n_jobs = 1

    x1 = f1.x()
    if f2 is None:
        x2 = x1
    else:
        x2 = f2.x()

    if f2 is not None:
        if batch_size is None:
            fs1, Ls1, _ = f1[:]
            fs2, Ls2, _ = f2[:]
            val1 = converter(x1, fs1, Ls1)
            val2 = converter(x2, fs2, Ls2)
            D = distfunc(val1, val2, n_jobs)
            logger("1/1")
        else:
            N1, N2 = len(f1), len(f2) if f2 is not None else len(f1)
            num_batches_1 = (N1 // batch_size) + int(bool(N1 % batch_size))
            num_batches_2 = (N2 // batch_size) + int(bool(N2 % batch_size))

            D = np.empty((N1, N2), dtype=np.float64)

            for i in range(num_batches_1):
                fs1, Ls1, _ = f1[i * batch_size : (i + 1) * batch_size]
                val1 = converter(x1, fs1, Ls1)
                for j in range(num_batches_2):
                    fs2, Ls2, _ = f2[j * batch_size : (j + 1) * batch_size]
                    val2 = converter(x2, fs2, Ls2)
                    D[
                        i * batch_size : (i + 1) * batch_size,
                        j * batch_size : (j + 1) * batch_size,
                    ] = distfunc(val1, val2, n_jobs)
                    logger(
                        f"{i * num_batches_2 + j + 1}/{num_batches_1 * num_batches_2}"
                    )
    else:
        if batch_size is None:
            fs1, Ls1, _ = f1[:]
            val1 = converter(x1, fs1, Ls1)
            D = distfunc(val1, None, n_jobs)
            logger("1/1")
        else:
            N = len(f1)
            num_batches = (N // batch_size) + int(bool(N % batch_size))

            D = np.empty((N, N), dtype=np.float64)

            for i in range(num_batches):
                fs1, Ls1, _ = f1[i * batch_size : (i + 1) * batch_size]
                val1 = converter(x1, fs1, Ls1)
                # diagonal
                D[
                    i * batch_size : (i + 1) * batch_size,
                    i * batch_size : (i + 1) * batch_size,
                ] = distfunc(val1, None, n_jobs)
                # off-diagonal
                for j in range(i + 1, num_batches):
                    fs2, Ls2, _ = f1[j * batch_size : (j + 1) * batch_size]
                    val2 = converter(x2, fs2, Ls2)
                    dist = distfunc(val1, val2, n_jobs)
                    D[
                        i * batch_size : (i + 1) * batch_size,
                        j * batch_size : (j + 1) * batch_size,
                    ] = dist
                    D[
                        j * batch_size : (j + 1) * batch_size,
                        i * batch_size : (i + 1) * batch_size,
                    ] = dist.T
                    logger(f"{i * num_batches + j + 1}/{num_batches ** 2}")
    return D


def _euclidean_converter(x, fs, Ls):
    return (x, fs)


def _euclidean_distfunc(val1, val2, n_jobs):
    x, fs1 = val1
    if val2 is None:
        fs2 = fs1
    else:
        _, fs2 = val2
    # TODO: implement more efficient algorithm for fs2==fs1
    diff = fs1[:, np.newaxis, :] - fs2[np.newaxis, :, :]
    D = np.sqrt(np.trapezoid(diff**2, x, axis=2))
    return D


def distmat_euclidean(f1, f2=None, batch_size=None, logger=lambda x: None):
    """L2 distance matrix between profiles.

    Parameters
    ----------
    f1 : heavyedge.ProfileData
        Open h5 file.
    f2 : heavyedge.ProfileData, optional
        Open h5 file.
        If not passed, it is set to *f1*.
    batch_size : int, optional
        Batch size to load data.
        If not passed, all data are loaded at once.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Returns
    -------
    (N1, N2) array
        Euclidean distance matrix.

    Notes
    -----
    ``distmat_euclidean(f1)`` is faster than ``distmat_euclidean(f1, f1)``.

    Examples
    --------
    >>> from heavyedge import ProfileData
    >>> from heavyedge_distance import get_sample_path
    >>> from heavyedge_distance.api import distmat_euclidean
    >>> with ProfileData(get_sample_path("MeanProfiles-AreaScaled.h5")) as data:
    ...     D = distmat_euclidean(data)
    """
    converter = _euclidean_converter
    distfunc = _euclidean_distfunc
    return _distmat(converter, distfunc, f1, f2, batch_size, logger)


def _wasserstein_converter(t):
    def converter(x, fs, Ls):
        Qs = quantile(x, fs, Ls, t)
        return Qs

    return converter


def _wasserstein_distfunc(t):
    def distfunc(Qs1, Qs2, n_jobs):
        D = wdist(t, Qs1, Qs2)
        return D

    return distfunc


def distmat_wasserstein(t, f1, f2=None, batch_size=None, logger=lambda x: None):
    """Wasserstein distance matrix between area-scaled profiles.

    Parameters
    ----------
    t : (M,) ndarray
        Coordinates of grids over which the quantile functions will be measured.
        Must be strictly increasing from 0 to 1.
    f1 : heavyedge.ProfileData
        Open h5 file of area-scaled profiles.
    f2 : heavyedge.ProfileData, optional
        Open h5 file of area-scaled profiles.
        If not passed, it is set to *f1*.
    batch_size : int, optional
        Batch size to load data.
        If not passed, all data are loaded at once.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Returns
    -------
    (N1, N2) array
        Wasserstein distance matrix.

    Notes
    -----
    ``distmat_wasserstein(f1)`` is faster than ``distmat_wasserstein(f1, f1)``.

    Examples
    --------
    >>> import numpy as np
    >>> from heavyedge import ProfileData
    >>> from heavyedge_distance import get_sample_path
    >>> from heavyedge_distance.api import distmat_wasserstein
    >>> with ProfileData(get_sample_path("MeanProfiles-AreaScaled.h5")) as data:
    ...     D = distmat_wasserstein(np.linspace(0, 1, 100), data)
    """
    converter = _wasserstein_converter(t)
    distfunc = _wasserstein_distfunc(t)
    return _distmat(converter, distfunc, f1, f2, batch_size, logger)


def _dfd_converter(x, fs, Ls):
    return (fs, Ls)


def _dfd_distfunc(val1, val2, n_jobs):
    return dfd(val1, val2, n_jobs)


def distmat_frechet(f1, f2=None, batch_size=None, n_jobs=None, logger=lambda x: None):
    """1-D discrete Fréchet distance matrix between profiles.

    Parameters
    ----------
    f1 : heavyedge.ProfileData
        Open h5 file.
    f2 : heavyedge.ProfileData, optional
        Open h5 file.
        If not passed, it is set to *f1*.
    batch_size : int, optional
        Batch size to load data.
        If not passed, all data are loaded at once.
    n_jobs : int, optional
        Number of parallel workers.
        If not passed, `HEAVYEDGE_MAX_WORKERS` environment variable is used.
        If the environment variable is invalid, set to 1.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Returns
    -------
    (N1, N2) array
        Discrete Fréchet distance matrix.

    Notes
    -----
    ``distmat_frechet(f1)`` is faster than ``distmat_frechet(f1, f1)``.

    Examples
    --------
    >>> from heavyedge import ProfileData
    >>> from heavyedge_distance import get_sample_path
    >>> from heavyedge_distance.api import distmat_frechet
    >>> with ProfileData(get_sample_path("MeanProfiles-PlateauScaled.h5")) as data:
    ...     D = distmat_frechet(data)
    """
    return _distmat(_dfd_converter, _dfd_distfunc, f1, f2, batch_size, n_jobs, logger)
