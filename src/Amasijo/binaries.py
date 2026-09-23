"""
Utilities for generating unresolved binary-system photometry.

When the binary module is active, every object is treated as a binary
system. The secondary-to-primary mass ratio is defined as

    q = M_secondary / M_primary

with

    0 < q <= 1.

The mass-ratio distribution is therefore responsible for determining
how similar the systems are to single stars. Very small q values
produce systems whose secondary contributes very little flux.

This module does not perform stellar-model interpolation. It only
handles mass-ratio generation and the combination of component
photometry.
"""

import numpy as np


def combine_magnitudes(mag1, mag2):
    """
    Combine the magnitudes of two unresolved sources.

    Parameters
    ----------
    mag1 : float or array-like
        Magnitude(s) of the first component.

    mag2 : float or array-like
        Magnitude(s) of the second component.

    Returns
    -------
    float or numpy.ndarray
        Magnitude(s) of the unresolved system.

    Notes
    -----
    The magnitudes are converted to relative fluxes, added, and
    converted back to magnitudes:

        m_system =
            -2.5 * log10(
                10**(-0.4*m1) +
                10**(-0.4*m2)
            )

    The two magnitudes must belong to the same photometric band and
    magnitude system.
    """
    mag1 = np.asarray(mag1, dtype=float)
    mag2 = np.asarray(mag2, dtype=float)

    return -2.5 * np.log10(
        np.power(10.0, -0.4 * mag1)
        + np.power(10.0, -0.4 * mag2)
    )


def generate_mass_ratios(
    n_stars,
    q_distribution="uniform",
    q_limits=(0.1, 1.0),
    random_state=None,
):
    """
    Generate mass ratios for a population of binary systems.

    Every object is treated as a binary system and therefore receives
    a strictly positive mass ratio.

    Parameters
    ----------
    n_stars : int
        Number of stellar systems.

    q_distribution : str, optional
        Probability distribution used to generate q.

        Currently supported:

        ``"uniform"``
            Uniform distribution between q_limits.

    q_limits : tuple of float, optional
        Minimum and maximum mass ratio:

            (q_min, q_max)

        Must satisfy

            0 < q_min <= q_max <= 1.

    random_state : int, numpy.random.RandomState,
                   numpy.random.Generator, or None, optional
        Random-number generator or seed.

    Returns
    -------
    q : numpy.ndarray
        Array of length ``n_stars`` containing one positive mass ratio
        for every stellar system.

    Notes
    -----
    There is deliberately no binary fraction in this function.

    Whether the binary module is active is decided by ``Amasijo``.
    If it is active, every generated system receives a secondary.

    The lower limit of q is therefore important: systems with small
    q are effectively close to single stars because their secondary
    contributes little flux.
    """

    if not isinstance(n_stars, (int, np.integer)):
        raise TypeError("n_stars must be an integer.")

    if n_stars < 0:
        raise ValueError("n_stars must be non-negative.")

    if len(q_limits) != 2:
        raise ValueError(
            "q_limits must contain exactly two values: "
            "(q_min, q_max)."
        )

    q_min, q_max = q_limits

    if not 0.0 < q_min <= q_max <= 1.0:
        raise ValueError(
            "q_limits must satisfy "
            "0 < q_min <= q_max <= 1."
        )

    if not isinstance(q_distribution, str):
        raise TypeError(
            "q_distribution must be a string."
        )

    if isinstance(random_state, np.random.Generator):
        rng = random_state

    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

    else:
        rng = np.random.default_rng(random_state)

    if q_distribution.lower() == "uniform":

        q = rng.uniform(
            q_min,
            q_max,
            size=n_stars,
        )

    else:
        raise ValueError(
            f"Unsupported q_distribution: '{q_distribution}'. "
            "Currently only 'uniform' is supported."
        )

    return q


def combine_photometry(
    primary,
    secondary,
    q,
    bands=None,
    suffix="_mag",
):
    """
    Combine primary and secondary photometry into unresolved
    binary-system photometry.

    Every row represents one binary system. There is no single-star
    case in this function: every q must satisfy 0 < q <= 1.

    Parameters
    ----------
    primary : pandas.DataFrame
        DataFrame containing the photometry of the primary components.

    secondary : pandas.DataFrame
        DataFrame containing the photometry of the secondary
        components.

    q : array-like
        Positive secondary-to-primary mass ratios.

    bands : tuple or list of str, optional
        Photometric bands to combine.

    suffix : str, optional
        Suffix used to construct photometric column names.

    Returns
    -------
    pandas.DataFrame
        Copy of ``primary`` containing the unresolved binary
        photometry.

    Raises
    ------
    TypeError
        If primary or secondary is not a pandas DataFrame.

    ValueError
        If the DataFrames do not have the same number of rows, if q
        has an incompatible length, or if any q <= 0 or q > 1.

    KeyError
        If a requested photometric column is missing.
    """
    import pandas as pd

    if not isinstance(primary, pd.DataFrame):
        raise TypeError(
            "primary must be a pandas.DataFrame."
        )

    if not isinstance(secondary, pd.DataFrame):
        raise TypeError(
            "secondary must be a pandas.DataFrame."
        )

    if len(primary) != len(secondary):
        raise ValueError(
            "primary and secondary must contain the same number "
            "of rows."
        )

    q = np.asarray(q, dtype=float)

    if q.ndim != 1:
        raise ValueError(
            "q must be a one-dimensional array."
        )

    if len(q) != len(primary):
        raise ValueError(
            "q must contain one value for every stellar system."
        )

    if np.any((q <= 0.0) | (q > 1.0)):
        raise ValueError(
            "All mass ratios must satisfy 0 < q <= 1."
        )

    combined = primary.copy()

    for band in bands:

        column = f"{band}{suffix}"

        if column not in primary.columns:
            raise KeyError(
                f"Column '{column}' not found in primary photometry."
            )

        if column not in secondary.columns:
            raise KeyError(
                f"Column '{column}' not found in secondary photometry."
            )

        combined[column] = combine_magnitudes(
            primary[column].to_numpy(),
            secondary[column].to_numpy(),
        )

    return combined