"""
Utilities for generating stellar-system photometry.

Amasijo treats every object as a stellar system consisting of a
primary component and an optional secondary component.

The secondary-to-primary mass ratio is defined as

    q = M_secondary / M_primary

with

    0 <= q <= 1.

A single star is represented by q = 0. In that case the secondary
component has zero flux and the system photometry is identical to
the primary photometry.

For q > 0, the secondary component is interpolated independently
from the stellar-evolution model and its flux is added to the
primary flux to obtain the unresolved system photometry.

This module does not perform stellar-model interpolation. It only
handles system parameters and the combination of component
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

    This function should not be called with an infinite or undefined
    secondary magnitude. For q = 0 systems, use
    ``combine_single_and_binary_magnitudes`` or handle the zero-flux
    secondary explicitly.
    """
    mag1 = np.asarray(mag1, dtype=float)
    mag2 = np.asarray(mag2, dtype=float)

    return -2.5 * np.log10(
        np.power(10.0, -0.4 * mag1)
        + np.power(10.0, -0.4 * mag2)
    )


def combine_single_and_binary_magnitudes(
    primary_mag,
    secondary_mag,
    q,
):
    """
    Combine primary and secondary magnitudes using the mass ratio.

    Systems with q = 0 are treated as single stars. For these systems
    the secondary contributes zero flux and the returned magnitude is
    exactly the primary magnitude.

    Parameters
    ----------
    primary_mag : float or array-like
        Magnitude(s) of the primary component.

    secondary_mag : float or array-like
        Magnitude(s) of the secondary component. Values corresponding
        to q = 0 are ignored.

    q : float or array-like
        Secondary-to-primary mass ratio(s).

        q = 0
            Single star.

        q > 0
            Unresolved binary.

    Returns
    -------
    float or numpy.ndarray
        Magnitude(s) of the unresolved system.

    Raises
    ------
    ValueError
        If any q value is outside the interval [0, 1].
    """
    primary_mag = np.asarray(primary_mag, dtype=float)
    secondary_mag = np.asarray(secondary_mag, dtype=float)
    q = np.asarray(q, dtype=float)

    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError(
            "Mass ratios must satisfy 0 <= q <= 1."
        )

    # Make scalar inputs and arrays behave consistently.
    primary_mag, secondary_mag, q = np.broadcast_arrays(
        primary_mag,
        secondary_mag,
        q,
    )

    # Start with the single-star solution. This also guarantees that
    # q = 0 returns the primary magnitude without any numerical
    # manipulation.
    combined_mag = primary_mag.copy()

    binary = q > 0.0

    if np.any(binary):
        combined_mag[binary] = combine_magnitudes(
            primary_mag[binary],
            secondary_mag[binary],
        )

    # Preserve scalar output when scalar inputs were supplied.
    if combined_mag.ndim == 0:
        return combined_mag.item()

    return combined_mag


def generate_mass_ratios(
    n_stars,
    binary_fraction=0.0,
    q_distribution="uniform",
    q_limits=(0.1, 1.0),
    random_state=None,
):
    """
    Generate mass ratios for a population of stellar systems.

    Every object is represented as a stellar system. Single stars are
    assigned q = 0, while binaries are assigned q > 0.

    Parameters
    ----------
    n_stars : int
        Number of stellar systems to generate.

    binary_fraction : float, optional
        Fraction of systems that are binaries.

        Systems not selected as binaries receive q = 0.

    q_distribution : str, optional
        Probability distribution used to generate q for binary
        systems.

        Currently supported:

        ``"uniform"``
            Uniform distribution between q_limits.

    q_limits : tuple of float, optional
        Minimum and maximum mass ratio for binary systems:

            (q_min, q_max)

        Must satisfy

            0 < q_min <= q_max <= 1.

        q = 0 is reserved for single stars and is therefore not
        included in q_limits.

    random_state : int, numpy.random.RandomState,
                   numpy.random.Generator, or None, optional
        Random-number generator or seed.

    Returns
    -------
    q : numpy.ndarray
        Array of length ``n_stars`` containing the mass ratio of every
        system.

        Single stars have q = 0.

        Binary systems have q > 0.

    Notes
    -----
    The binary fraction controls how many systems receive q > 0.
    It does not modify the distribution of q among the binary
    systems.
    """
    if not isinstance(n_stars, (int, np.integer)):
        raise TypeError("n_stars must be an integer.")

    if n_stars < 0:
        raise ValueError("n_stars must be non-negative.")

    if not 0.0 <= binary_fraction <= 1.0:
        raise ValueError(
            "binary_fraction must be between 0 and 1."
        )

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

    # Create or use the requested random-number generator.
    if isinstance(random_state, np.random.Generator):
        rng = random_state

    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

    else:
        rng = np.random.default_rng(random_state)

    # Initialize every system as a single star.
    q = np.zeros(n_stars, dtype=float)

    # Select systems that will contain a secondary.
    is_binary = rng.random(n_stars) < binary_fraction

    n_binaries = np.count_nonzero(is_binary)

    if n_binaries == 0:
        return q

    # Generate mass ratios for binary systems.
    if q_distribution.lower() == "uniform":
        q[is_binary] = rng.uniform(
            q_min,
            q_max,
            size=n_binaries,
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
    bands=("G", "BP", "RP"),
    suffix="_mag",
):
    """
    Combine component photometry into unresolved system photometry.

    Every row represents one stellar system. Systems with q = 0 are
    single stars and retain the primary photometry. Systems with
    q > 0 are unresolved binaries and have the primary and secondary
    fluxes combined.

    Parameters
    ----------
    primary : pandas.DataFrame
        DataFrame containing the photometry of the primary components.

    secondary : pandas.DataFrame
        DataFrame containing the photometry of the secondary
        components.

        For q = 0 systems, the secondary photometry does not need to
        correspond to a physical star and is ignored.

    q : array-like
        Mass ratio of each system.

        q = 0 identifies single-star systems.

    bands : tuple or list of str, optional
        Photometric bands to combine.

    suffix : str, optional
        Suffix used to construct photometric column names.

    Returns
    -------
    pandas.DataFrame
        Copy of ``primary`` containing the unresolved system
        photometry.

    Raises
    ------
    TypeError
        If primary or secondary is not a pandas DataFrame.

    ValueError
        If the DataFrames do not have the same number of rows or if
        q has an incompatible length.

    KeyError
        If a requested photometric column is missing.

    Notes
    -----
    This function deliberately processes all systems through the same
    interface. A single star is simply the q = 0 special case.
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

    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError(
            "Mass ratios must satisfy 0 <= q <= 1."
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

        combined[column] = combine_single_and_binary_magnitudes(
            primary[column].to_numpy(),
            secondary[column].to_numpy(),
            q,
        )

    return combined