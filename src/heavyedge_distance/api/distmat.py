import os

import numpy as np

from heavyedge_distance.dfd import dfd

__all__ = [
    "distmat_frechet",
]


def distmat_frechet(f1, f2=None, batch_size=None, n_jobs=None, logger=None):
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
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_distance.api import distmat_frechet
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     D1 = distmat_frechet(data)
    """
    if n_jobs is not None:
        pass
    else:
        n_jobs = os.environ.get("HEAVYEDGE_MAX_WORKERS")
        if n_jobs is not None:
            n_jobs = int(n_jobs)
        else:
            n_jobs = 1

    if logger is None:
        # dummy logger
        def logger(msg):
            pass

    if f2 is not None:
        # distmat between f1 and f2
        if batch_size is None:
            Ys1, Ls1, _ = f1[:]
            Ys2, Ls2, _ = f2[:]
            D = dfd((Ys1, Ls1), (Ys2, Ls2), n_jobs)
            logger("1/1")
        else:
            N1, N2 = len(f1), len(f2) if f2 is not None else len(f1)
            num_batches_1 = (N1 // batch_size) + int(bool(N1 % batch_size))
            num_batches_2 = (N2 // batch_size) + int(bool(N2 % batch_size))

            D = np.empty((N1, N2), dtype=np.float64)

            for i in range(num_batches_1):
                Ys1, Ls1, _ = f1[i * batch_size : (i + 1) * batch_size]
                for j in range(num_batches_2):
                    Ys2, Ls2, _ = f2[j * batch_size : (j + 1) * batch_size]
                    D[
                        i * batch_size : (i + 1) * batch_size,
                        j * batch_size : (j + 1) * batch_size,
                    ] = dfd((Ys1, Ls1), (Ys2, Ls2), n_jobs)
                    logger(
                        f"{i * num_batches_2 + j + 1}/{num_batches_1 * num_batches_2}"
                    )
    else:
        # distmat of f1 to itself
        if batch_size is None:
            Ys1, Ls1, _ = f1[:]
            D = dfd((Ys1, Ls1), None, n_jobs)
            logger("1/1")
        else:
            N = len(f1)
            num_batches = (N // batch_size) + int(bool(N % batch_size))

            D = np.empty((N, N), dtype=np.float64)

            for i in range(num_batches):
                Ys1, Ls1, _ = f1[i * batch_size : (i + 1) * batch_size]
                # diagonal
                D[
                    i * batch_size : (i + 1) * batch_size,
                    i * batch_size : (i + 1) * batch_size,
                ] = dfd((Ys1, Ls1), None, n_jobs)
                # off-diagonal
                for j in range(i + 1, num_batches):
                    Ys2, Ls2, _ = f1[j * batch_size : (j + 1) * batch_size]
                    dist = dfd((Ys1, Ls1), (Ys2, Ls2), n_jobs)
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
