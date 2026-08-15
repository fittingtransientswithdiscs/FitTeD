"""kerrgeo.thermal — Planck weights for energy-resolved Stokes integration.

This module provides the dimensionless Planck specific-intensity helper
``planck_B(E, T)`` and the per-pixel observer-frame weight
``planck_weight(E_obs, T_eff, g, f_col)`` used by the energy-resolved
path of ``reflected_polarized_trace``.

In the Thomson-dominated regime that ``ThermalDisk`` assumes, both
Chandrasekhar limb polarization and Rayleigh reflection are achromatic
in the local rest frame, so the geometry-only (Q/I, U/I) per-pixel
factorization computed by the existing pipeline can be re-used for any
observer-energy slice.  This module supplies the per-pixel intensity
weight that converts that factorization into an observer specific
intensity ``I_E(E_obs)``.

Conventions
-----------
Both ``E`` and ``T`` carry the user's choice of units (K, keV, or
dimensionless — anything works, as long as both are in the same units).
Internally only the ratio ``x = E/T`` enters Planck's law, so units
cancel out.  This lets users work either:

* in dimensionless mode with ``T`` normalized so peak T_max = 1 and
  ``E_obs`` in those same dimensionless units (good for shape-only
  diagnostic studies, e.g. PD vs. ``E_obs/T_max``); or
* in physical mode with ``T`` in K (or keV) and ``E_obs`` in matching
  units, so that radius-dependent color-correction models like
  Davis–Hubeny that need an absolute temperature can be wired in.

Normalization
-------------
We use

    B_E(T) = (15/π⁴) · E³ / (exp(E/T) − 1)

with the prefactor chosen so that

    ∫_0^∞ B_E(T) dE = T⁴

(Stefan–Boltzmann in the user's units).  This makes the bolometric back
compatibility test (energy-resolved → bolometric reduces to ε(r) ∝ T(r)⁴)
drop out exactly, and matches the convention used by ``ThermalDisk``.

Energy-resolved observer-frame specific intensity
-------------------------------------------------
For a Thomson-thick atmosphere with effective temperature T_eff(r) and
color correction f_col(r), the local emergent specific intensity in the
fluid rest frame is the Shimura–Takahara diluted Planck

    I_E(local) = B_{E_em}(f_col · T_eff) / f_col⁴.

The dilution factor 1/f_col⁴ conserves bolometric flux — integrating
over E_em recovers ε(r) = T_eff(r)⁴.

The observer at infinity sees specific intensity

    I_{E_obs}(observer) = g³ · I_E(local) |_{E_em = E_obs/g}

(the Lorentz invariant is I_E / E³, so I_E,obs = g³ · I_E,em at fixed
ν_obs).  Combining, the per-pixel Planck weight is

    planck_weight(E_obs, T_eff, g, f_col)
        = g³ · B_{E_obs/g}(f_col · T_eff) / f_col⁴.

For the reflected channel, ``g`` is the chained product
``g_leg1 · g_leg2`` because elastic Thomson scattering preserves the
photon energy in the landing fluid rest frame.

References
----------
* Shimura & Takahara 1995, ApJ 445, 780 — color correction for
  thermal-disk atmospheres.
* Davis et al. 2005 ApJ 621, 372 / Done et al. 2012 MNRAS 420, 1848 —
  radius-dependent f_col(T_eff) models that motivate the callable
  f_col API.
"""

import numpy as np


__all__ = [
    "PLANCK_NORM",
    "planck_B",
    "planck_weight",
]


# Stefan–Boltzmann normalization: ∫₀^∞ x³/(exp(x)−1) dx = π⁴/15, so
# multiplying B by this constant gives ∫B_E(T) dE = T⁴.
PLANCK_NORM = 15.0 / (np.pi ** 4)


def planck_B(E, T):
    """Dimensionless Planck specific intensity B_E(T).

        B_E(T) = (15/π⁴) · E³ / (exp(E/T) − 1)

    Normalized so ∫₀^∞ B_E(T) dE = T⁴.  Both arguments must carry the
    same units; only the ratio E/T enters the body of Planck's law.

    Parameters
    ----------
    E : array_like
        Photon energy (any units).
    T : array_like
        Local temperature (same units as ``E``).

    Returns
    -------
    B : ndarray, broadcast shape of (E, T)
        Specific intensity in the convention above.  Returns 0 where
        T ≤ 0, E ≤ 0, or either is non-finite.

    Notes
    -----
    The implementation is numerically stable across the full range
    E/T ∈ (0, ∞):

      * For E/T < 50 we use ``np.expm1`` so the small-argument
        (Rayleigh–Jeans) regime is computed without catastrophic
        cancellation;
      * For E/T ≥ 50 we factor exp(−E/T) into the numerator so the
        Wien tail E³ · exp(−E/T) / (1 − exp(−E/T)) does not overflow.
    """
    E = np.asarray(E, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    valid = (
        np.isfinite(E) & np.isfinite(T) & (E > 0.0) & (T > 0.0)
    )
    E_safe = np.where(valid, E, 1.0)
    T_safe = np.where(valid, T, 1.0)
    x = E_safe / T_safe

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        # Numerically-stable two-branch form:
        #   small (x < 50): use expm1 → handles RJ limit cleanly.
        #   large (x ≥ 50): factor exp(-x) → handles Wien tail without overflow.
        small = (E_safe ** 3) / np.expm1(x)
        large = (E_safe ** 3) * np.exp(-x) / (1.0 - np.exp(-x))
    factor = np.where(x < 50.0, small, large)
    result = PLANCK_NORM * factor
    return np.where(valid, result, 0.0)


def planck_weight(E_obs, T_eff, g, f_col):
    """Observer-frame Planck weight for an energy-resolved Stokes pixel.

    Returns

        g³ · B_{E_obs/g}(f_col · T_eff) / f_col⁴ ,

    the observer-frame specific intensity contribution from a Thomson-
    dominated thermal patch with effective temperature T_eff and
    color-correction f_col, viewed through redshift g.

    For the direct channel, ``g`` is the camera (back-trace) redshift
    g_camera = ν_obs / ν_em.  For the reflected channel, ``g`` is the
    chained product g_leg1 · g_leg2 (elastic Thomson scattering
    preserves ω in the landing rest frame, so the two boost factors
    multiply through to a single net redshift).

    Parameters
    ----------
    E_obs : array_like
        Observer-frame photon energy.  Same units as ``T_eff``.
    T_eff : array_like
        Local emitter effective temperature (rest frame), same units
        as ``E_obs``.
    g : array_like
        Photon redshift factor (dimensionless, > 0).  Single leg for
        direct, chained product for reflected.
    f_col : array_like
        Color-correction factor at the emitter (dimensionless, > 0;
        typically ≈ 1.7 for hot disk atmospheres, optionally a
        function of T_eff and r).

    Returns
    -------
    weight : ndarray, broadcast shape of inputs.
        Per-pixel observer-frame specific intensity weight.  Returns 0
        wherever any input is non-physical (g ≤ 0, f_col ≤ 0, T ≤ 0,
        E_obs ≤ 0, or any non-finite).

    Notes
    -----
    The bolometric integral over E_obs reproduces the existing g⁴
    weighting used by the bolometric pipeline:

        ∫_0^∞ planck_weight(E, T, g, f_col) dE
            = g³ · ∫_0 ^∞ B_{E_em}(f_col·T) dE_obs / f_col⁴
            = g³ · g · (f_col·T)⁴ / f_col⁴            (subst. E_em = E_obs/g)
            = g⁴ · T⁴.

    This is the Shimura–Takahara color-corrected analogue of the
    Stefan–Boltzmann ε(r) ∝ T(r)⁴ relation, and is the basis of the
    bolometric backwards-compatibility regression test for the
    energy-resolved orchestrator.
    """
    E_obs = np.asarray(E_obs, dtype=np.float64)
    T_eff = np.asarray(T_eff, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    f_col = np.asarray(f_col, dtype=np.float64)

    valid = (
        np.isfinite(E_obs) & np.isfinite(T_eff)
        & np.isfinite(g) & np.isfinite(f_col)
        & (E_obs > 0.0) & (T_eff > 0.0)
        & (g > 0.0) & (f_col > 0.0)
    )
    g_safe = np.where(valid, g, 1.0)
    f_col_safe = np.where(valid, f_col, 1.0)
    T_eff_safe = np.where(valid, T_eff, 1.0)
    E_obs_safe = np.where(valid, E_obs, 1.0)

    E_em = E_obs_safe / g_safe
    T_col = f_col_safe * T_eff_safe

    B = planck_B(E_em, T_col)
    weight = (g_safe ** 3) * B / (f_col_safe ** 4)
    return np.where(valid, weight, 0.0)
