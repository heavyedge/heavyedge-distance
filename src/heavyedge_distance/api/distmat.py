import os

__all__ = [
    "distmat_frechet",
]


def distmat_frechet(f1, f2=None, batch_size=None, n_jobs=None):
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
    ...     D = distmat_frechet(data)
    """
    if n_jobs is not None:
        pass
    else:
        n_jobs = os.environ.get("HEAVYEDGE_MAX_WORKERS")
        if n_jobs is not None:
            n_jobs = int(n_jobs)
        else:
            n_jobs = 1
