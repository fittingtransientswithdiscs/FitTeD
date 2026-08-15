"""Pure-Python ray-traced disc SED — drop-in replacement for the
Fortran ``fitted.models.numerical_disc`` module.

Exposes the same names as ``numerical_disc.__init__``:

  * ``numerical_disc_model(bh_a, rout, incl, kTdisc_array, rdisc_array)``
  * ``energy_grid``                   — log-spaced, length nex+1 (=2049)
  * ``energy_grid_midpoints``         — bin widths dE_i, length nex (=2048)
  * ``numerical_disc_avaliable``      — string "Yes"/"No"
  * ``N``                             — astropy Quantity normalisation

Internally, the per-pixel disc SED is built from

  * ``..raytrace.intensity_trace`` — observer→disc photon back-trace plus the
    per-pixel redshift factor g = ω_obs / ω_em (closed-form Kerr, all
    elliptic), from the ray tracer vendored at ``fitted/models/raytrace``;
  * a Shimura–Takahara colour-corrected observer-frame Planck weight per
    (E_obs, T_eff, g, f_col), evaluated inline in the hot loop rather than via
    ``raytrace.thermal.planck_weight`` (see ``numerical_disc_model``).

``raytrace`` is a minimal extracted subset of the ``kerrgeo`` ray tracer,
verified bit-identical to it over a grid of spins, inclinations and source
types.  See ``fitted/models/raytrace/PROVENANCE.md``.

Per-pixel quadrature: a polar (ρ, φ) image-plane grid identical in
construction to the Fortran ``impactgrid`` (log-spaced ρ from
``rfunc(a, μ_0)`` to ``rout``, uniform φ over [0, 2π], β-axis squashed
by ``mueff = max(μ_0, 0.3)``).  This makes the resulting SEDs match the
Fortran reference to better than 1 % across the disc spectrum at
nro × nphi = 200 × 200 (the Fortran default).

Energy axis: the Fortran's internal log-spaced grid (Emin=1e-4 keV,
Emax=1e2 keV, nex=2**11) is reproduced exactly here.

Returned units: ``N.value × photar`` where photar is the bin-integrated
photon count per energy bin, matching the existing wrapper convention
that ``gr_disc_plus.get_Spectrum`` consumes.

Notes
-----
* ``incl`` is degrees (matches the Fortran ``param(3)``).
* ``rdisc_array`` is in r_g; passing a non-uniform grid is fine here
  (we use ``np.interp`` rather than the Fortran's uniform-grid index
  arithmetic), but ``gr_disc_plus`` already pre-pads to a uniform grid
  so this is moot in production.
* ``kTdisc_array`` is in keV (rest-frame effective temperature; the
  colour correction is applied internally per the same
  Done+ 2012 prescription as the Fortran ``fcol``).
* The colour correction is the Done+ 2012 piecewise law from
  ``tdedisc_grid.f90 :: function fcol``, evaluated at the local
  rest-frame T_eff (keV) — re-implemented in Python so this module has
  no Fortran dependency.
"""

from warnings import warn

import numpy as np
from astropy import constants, units

# Re-use the same physical normalisation constant as the Fortran wrapper
# so callers can swap modules without rescaling the output downstream.
normalisation_unit = (units.keV * units.M_sun ** -2 * units.s ** -1)
r_g = constants.G / constants.c ** 2
N = (r_g ** 2 * constants.sigma_sb * constants.k_B ** -4
     * units.keV ** 4).to(normalisation_unit)


# ---------------------------------------------------------------------------
# Energy grid — must match Fortran's ``internal_grids`` (nex = 2**11)
# ---------------------------------------------------------------------------
NEX = 2 ** 11             # 2048 fine bins (Fortran ``earx``)
NEC = 300                 # coarse bins (Fortran ``earc``) — see Notes below
_EMIN = 1e-4              # keV
_EMAX = 1e2               # keV

# Fine grid (matches Fortran's ``earx``)
_log_edges = np.linspace(0.0, 1.0, NEX + 1)
energy_grid = _EMIN * (_EMAX / _EMIN) ** _log_edges                   # length 2049
_E_mid = 0.5 * (energy_grid[1:] + energy_grid[:-1])                    # length 2048
energy_grid_midpoints = energy_grid[1:] - energy_grid[:-1]             # length 2048
# (Same name as the Fortran wrapper exposes — see numerical_disc/__init__.py
# line 19-20: ``energy_grid_midpoints = tdedisc_grid.internal_grids.dEarr``.
# It's the bin-width array, not the bin midpoints; we keep the legacy name
# for drop-in compatibility.)
_dE = energy_grid_midpoints

# Coarse grid (matches Fortran's ``earc``).  The Planck-weight inner sum
# is evaluated here (300 energies × N_pix exponentials) and then linearly
# interpolated onto the fine grid for the returned ``photar``.  This is
# the same coarse → fine quadrature trick the Fortran uses (``myinterp``
# in tdedisc_grid.f90), and gives a ~7× speedup at <1% spectral error.
_log_edges_c = np.linspace(0.0, 1.0, NEC + 1)
_earc = _EMIN * (_EMAX / _EMIN) ** _log_edges_c                       # length NEC+1
_E_mid_c = 0.5 * (_earc[1:] + _earc[:-1])                              # length NEC


# ---------------------------------------------------------------------------
# Done+ 2012 colour correction (Python clone of tdedisc_grid.f90 :: fcol)
# ---------------------------------------------------------------------------
def _fcol_done2012(T_keV):
    """Colour-correction f_col(T) at rest-frame T_eff in keV.

    Piecewise:
        f_col = 1                          (T < 2.585e-3 keV)
        f_col = (T / 2.585e-3)**0.833      (2.585e-3 ≤ T < 8.617e-3)
        f_col = (72 / T)**(1/9)            (T ≥ 8.617e-3)

    Bit-identical (modulo float32→float64) to the Fortran implementation.
    """
    T = np.asarray(T_keV, dtype=np.float64)
    out = np.ones_like(T)
    low = T < 2.585e-3
    mid = (T >= 2.585e-3) & (T < 8.617e-3)
    high = T >= 8.617e-3
    out[low] = 1.0
    out[mid] = (T[mid] / 2.585e-3) ** 0.833
    out[high] = (72.0 / T[high]) ** (1.0 / 9.0)
    out[~np.isfinite(T) | (T <= 0.0)] = 1.0
    return out


# Kelvin → keV conversion factor.  k_B / e (in keV).
_KELVIN_TO_KEV = 8.617333262e-8


def _apply_user_colour_correction(colour_correction, kT_keV, r_em):
    """Evaluate a user-supplied f_col(T_kelvin, r_rg) on per-pixel arrays.

    The user's callable takes T in Kelvin and r in r_g; numerical_disc_py
    natively carries kT in keV.  This helper handles the unit conversion
    and returns a 1-D array of f_col values, one per pixel.

    If ``colour_correction`` is None, falls back to the bit-identical
    legacy Done+ 2012 path used by the Fortran reference and the existing
    FitTeD baseline.
    """
    if colour_correction is None:
        return _fcol_done2012(kT_keV)
    T_kelvin = kT_keV / _KELVIN_TO_KEV
    fc = colour_correction(T_kelvin, r_em)
    return np.asarray(fc, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fortran-style polar impact-parameter grid (impactgrid + rfunc clones)
# ---------------------------------------------------------------------------
def _rfunc(a, mu0):
    """Inner edge of the impact-parameter grid, ρ_min(a, μ_0).

    Empirical fit calibrated by the FitTeD authors; reproduced
    bit-for-bit from ``tdedisc_grid.f90 :: function rfunc`` so the
    Python and Fortran pipelines start from the same innermost ray.

    Fortran reference (only TWO spin branches and a μ_0-linear cap
    that pulls ρ_min inward at high inclination):

        if a > 0.8:
            rfunc = 1.5 + 0.5 * mu0**5.5
            rfunc = min(rfunc, -0.1 + 5.6  * mu0)
            rfunc = max(0.1, rfunc)
        else:
            rfunc = 3.0 + 0.5 * mu0**5.5
            rfunc = min(rfunc, -0.2 + 10.0 * mu0)
            rfunc = max(0.1, rfunc)
    """
    if a > 0.8:
        r = 1.5 + 0.5 * mu0 ** 5.5
        r = min(r, -0.1 + 5.6 * mu0)
    else:
        r = 3.0 + 0.5 * mu0 ** 5.5
        r = min(r, -0.2 + 10.0 * mu0)
    return max(0.1, r)


def _impact_grid(rmin, rmax, mu0, nro=200, nphi=200):
    """Return (alpha, beta, dOmega) on a Fortran-equivalent polar grid.

    Uses log-spaced ρ from ``rmin`` to ``rmax`` and uniform φ over
    ``(0, 2π)``.  ``mueff = max(mu0, 0.3)`` squashes the β axis.

    Pixel solid-angle element (in the Cartesian (α, β) image plane,
    Jacobian-correct):

        dΩ(i, j) = ρ̄_i · (ρ_i − ρ_{i−1}) · mu_eff · 2π / nphi.
    """
    mueff = max(mu0, 0.3)
    log_edges = np.linspace(np.log10(rmin), np.log10(rmax), nro + 1)
    r_edges = 10.0 ** log_edges                       # (nro+1,)
    r_centres = 0.5 * (r_edges[1:] + r_edges[:-1])    # (nro,)
    dr = r_edges[1:] - r_edges[:-1]                   # (nro,)
    dphi = 2.0 * np.pi / nphi
    phi = (np.arange(nphi) + 0.5) * dphi              # (nphi,)

    R, PHI = np.meshgrid(r_centres, phi, indexing='ij')   # (nro, nphi)
    DR, _ = np.meshgrid(dr, phi, indexing='ij')

    alpha = R * np.sin(PHI)
    beta = R * np.cos(PHI) * mueff
    dOmega = R * DR * mueff * dphi
    return alpha, beta, dOmega


# ---------------------------------------------------------------------------
# Vendored ray tracer.  Ships with FitTeD, so this normally always succeeds;
# the guard exists so a damaged install degrades to a clear message rather
# than an import traceback.
# ---------------------------------------------------------------------------
numerical_disc_avaliable = "No"
try:
    from .. import raytrace as _raytrace  # noqa: F401
except ImportError:
    warn(
        "the vendored ray tracer (fitted.models.raytrace) could not be "
        "imported — numerical_disc_py cannot be used.  Reinstall FitTeD, or "
        "use the Fortran-backed numerical_disc instead.",
        stacklevel=2,
    )
else:
    numerical_disc_avaliable = "Yes"


# ---------------------------------------------------------------------------
# Minimum |a| the vendored ray tracer supports.
#
# The vectorised batch path does not dispatch to a Schwarzschild backend, so
# a = 0 exactly raises NotImplementedError.  Flooring |a| here is physically
# indistinguishable: measured over inclinations 30–85 deg, a = 1e-8 versus
# a = 1e-6 differ by <= 3e-6 in both g and r_hit and give an identical
# validity mask.  Retrograde spins keep their sign.
# See fitted/models/raytrace/PROVENANCE.md.
# ---------------------------------------------------------------------------
_A_MIN = 1e-8


def _floor_spin(a):
    """Return ``a`` with ``|a|`` floored at ``_A_MIN``, preserving sign."""
    a = float(a)
    if abs(a) < _A_MIN:
        return _A_MIN if a >= 0.0 else -_A_MIN
    return a


# ---------------------------------------------------------------------------
# Per-(a, incl, rout, grid) trace cache.
#
# In a typical FitTeD MCMC run, get_Spectrum is called ~10³–10⁴ times with
# the same (a, incl) but different DiscT profiles per epoch.  The geodesic
# trajectory in Kerr does not depend on photon energy or local emissivity,
# so the per-pixel (r_em, g, dΩ, valid) tuple is purely a function of
# (bh_a, incl, rout, nro, nphi, r_obs).  Cache it.
#
# Limited to MAX_TRACE_CACHE entries (LRU-ish via insertion-order dict pop)
# to avoid unbounded memory growth if a sampler walks (a, incl) finely.
# ---------------------------------------------------------------------------
_TRACE_CACHE = {}
MAX_TRACE_CACHE = 32


def clear_trace_cache():
    """Drop the cached (a, incl) traces.  Useful in long-running notebooks
    or when memory pressure becomes an issue."""
    _TRACE_CACHE.clear()


def _cached_trace(bh_a, incl_rad, rout, nro, nphi, r_obs,
                  source_type='keplerian'):
    """Return (r_em_flat, g_flat, dOmega_flat, valid) for a (a, incl, rout)
    triple, computing it on miss and caching the result.

    The arrays are flattened to 1-D and pre-masked to keep only the
    pixels that landed on the disc — this is what the per-call hot loop
    consumes, so we never re-do the masking either.

    Parameters
    ----------
    source_type : str
        Which raytrace emitter source to use:
          ``'keplerian'`` — circular orbits only; intra-ISCO pixels NaN'd.
                            Suitable for TDE-like discs that truncate at r_I.
          ``'kerr_disk'`` — proper plunging-geodesic continuation inside the
                            ISCO (E_ISCO, L_ISCO conserved + u^r from u·u = -1).
                            Required for hot-gas emission inside the ISCO,
                            cf. Mummery, Ingram, Davis & Fabian (2024).
    """
    if source_type not in ('keplerian', 'kerr_disk'):
        raise ValueError(f"unknown source_type {source_type!r}")
    key = (float(bh_a), float(incl_rad), float(rout),
           int(nro), int(nphi), float(r_obs), source_type)
    if key in _TRACE_CACHE:
        # Touch (re-insert) so it stays warm under the size cap.
        val = _TRACE_CACHE.pop(key)
        _TRACE_CACHE[key] = val
        return val

    from ..raytrace import (
        AsymptoticObserver, EquatorialDisk, KeplerianDisk, intensity_trace,
    )

    mu0 = float(np.cos(incl_rad))
    rmin = _rfunc(bh_a, mu0)
    alpha, beta, dOmega = _impact_grid(rmin, float(rout), mu0,
                                        nro=nro, nphi=nphi)

    obs = AsymptoticObserver(r_obs=float(r_obs), inclination=float(incl_rad))
    disk = EquatorialDisk()
    if source_type == 'keplerian':
        src = KeplerianDisk(a=float(bh_a), prograde=True)
    else:
        from ..raytrace import KerrDisk
        src = KerrDisk(a=float(bh_a), prograde=True)
    out = intensity_trace(a=float(bh_a), observer=obs, target=disk, source=src,
                          pixels=dict(alpha=alpha, beta=beta))

    valid = out['valid']
    r_em_flat = out['r_hit'][valid]
    g_flat = out['g'][valid]
    dOmega_flat = dOmega[valid]

    val = (r_em_flat, g_flat, dOmega_flat, valid)
    _TRACE_CACHE[key] = val
    if len(_TRACE_CACHE) > MAX_TRACE_CACHE:
        # Pop the oldest (Python ≥3.7 dicts are insertion-ordered).
        first_key = next(iter(_TRACE_CACHE))
        _TRACE_CACHE.pop(first_key)
    return val


__all__ = [
    "numerical_disc_model",
    "numerical_disc_dNdE_at_vs",
    "energy_grid",
    "energy_grid_midpoints",
    "numerical_disc_avaliable",
    "N",
]


# ---------------------------------------------------------------------------
# The drop-in model function
# ---------------------------------------------------------------------------
_E_mid_c_sq = _E_mid_c ** 2  # cached: per-energy E_obs² factor


def _auto_grid(bh_a, incl_deg):
    """Pick (nro, nphi) adaptively for ≤1e-3 SED accuracy.

    Calibration grid (300×300 reference, kT_peak = 6e5 K disc, mask
    cuts on bins ≥1e-3 of peak photar, run on a representative TDE
    profile); see ``outputs/pixel_convergence.py``.

    Inclination is the dominant driver (Doppler-boosted limb structure
    sharpens as i → 90°).  Spin matters only at the very-high-spin and
    near-edge-on corner.
    """
    a = float(abs(bh_a))
    incl = float(incl_deg)
    if incl >= 82.0 and a >= 0.9:
        return 240, 240        # extreme corner: needs ~58k pixels
    if incl >= 78.0:
        return 200, 200        # near edge-on: matches Fortran default
    if incl >= 65.0:
        return 140, 140
    return 100, 100             # face-on to moderate: 10k pixels enough


def numerical_disc_model(bh_a, rout, incl, kTdisc_array, rdisc_array,
                         nro=None, nphi=None, r_obs=1.0e4,
                         dtype=np.float64, colour_correction=None):
    """Observer-frame thermal-disc SED on the internal energy grid.

    Drop-in replacement for ``numerical_disc.numerical_disc_model``.

    Parameters
    ----------
    bh_a : float
        BH spin parameter (M = 1 units).
    rout : float
        Outer ray-tracing boundary in r_g (matches Fortran ``param(2)``).
        Inner edge is r_isco(a) (Bardeen prograde).
    incl : float
        Disc–observer inclination, **degrees** (matches Fortran).
    kTdisc_array : (nr,) array_like
        Per-radius rest-frame effective temperature, **keV**.
    rdisc_array : (nr,) array_like
        Matching radii, **r_g**.
    nro, nphi : int or None, optional
        Image-plane (radial, azimuthal) pixel counts.  ``None`` (default)
        picks an adaptive grid based on (bh_a, incl) calibrated for ≤1e-3
        SED accuracy at production TDE temperatures (see ``_auto_grid``).
        Pass explicit integers to override — e.g. ``nro=nphi=200`` to
        match the Fortran's fixed default.
    r_obs : float, optional
        Asymptotic observer radius in r_g.  Default 1e4 — large enough
        that the static-observer 4-velocity is asymptotic to within
        O(M / r_obs) = O(1e-4).
    dtype : np.dtype, optional
        Precision for the (NEC, n_pix) inner broadcast.  ``np.float64``
        (default) gives bit-identical results to the legacy code path
        (tested 1.3e-10 bolometric agreement).  ``np.float32`` saves
        ~25 % runtime on top, with bolometric error ~3e-8 — well below
        any astrophysical noise floor.  Use float32 for MCMC inner-loop
        sampling on large pixel grids.

    Returns
    -------
    photar : ndarray, shape (1, nex)
        Observer-frame photon-count SED on the internal energy grid.
        The leading length-1 axis matches the Fortran wrapper output shape.

    Notes
    -----
    The inner-loop integrand uses an algebraic identity that cancels the
    geodesic redshift factor ``g`` against the corresponding factor in
    the Lorentz-invariant ``I_E / E³``::

        B(E_em, T_col) · pixel_w / E_obs
            = [norm · E_em³ / expm1(x)] · [g³ · dΩ / f⁴] / E_obs
            = norm · E_obs² · dΩ / [f⁴ · expm1(E_obs / (g·f·T))]

    so the only quantity that depends on both the energy grid and the
    pixel index is the dimensionless ``x = E_obs / (g · f · T)``.  We
    avoid building the intermediate ``E_em``, ``E_em^3`` and ``B``
    arrays entirely, cutting the hot-path runtime by ~9× vs a direct
    application of ``planck_weight`` on broadcasted (NEC × n_pix) inputs.
    """
    if numerical_disc_avaliable != "Yes":
        raise NotImplementedError(
            "the vendored ray tracer (fitted.models.raytrace) is not "
            "importable; reinstall FitTeD."
        )

    bh_a = _floor_spin(bh_a)
    incl_deg = float(incl)
    incl_rad = np.deg2rad(incl_deg)

    if nro is None or nphi is None:
        nro_auto, nphi_auto = _auto_grid(bh_a, incl_deg)
        if nro is None: nro = nro_auto
        if nphi is None: nphi = nphi_auto

    kTdisc_array = np.asarray(kTdisc_array, dtype=np.float64)
    rdisc_array = np.asarray(rdisc_array, dtype=np.float64)
    if kTdisc_array.shape != rdisc_array.shape:
        raise ValueError("kTdisc_array and rdisc_array must have the same length")

    # ---- 1+2. Geodesic-only quantities, cached per (a, incl, rout, grid) ----
    # The trace is the expensive step; in MCMC fitting only the temperature
    # profile changes between calls, so we re-use the cached (r_em, g, dΩ).
    r_em, g, dO_flat, _valid = _cached_trace(
        bh_a, incl_rad, float(rout), int(nro), int(nphi), float(r_obs),
    )

    if g.size == 0:
        return (N.value * np.zeros(NEX))[None, :]

    # ---- 3. Local rest-frame T_eff and colour correction ----
    # Linear interpolation matches the Fortran's intra-bin behaviour for
    # uniform input grids; np.interp also handles non-uniform input.
    kT_flat = np.interp(r_em, rdisc_array, kTdisc_array,
                        left=0.0, right=0.0)
    kT_flat = np.where(kT_flat > 0.0, kT_flat, 0.0)
    # Apply user-supplied f_col(T_kelvin, r_rg) if given; otherwise
    # use the bit-identical legacy Done+ 2012 path.
    fc_flat = _apply_user_colour_correction(colour_correction, kT_flat, r_em)

    # Mask out pixels with zero temperature (outside the disc support);
    # they contribute nothing and we'd waste broadcast memory carrying
    # them through the (NEC, n_pix) integrand.
    Tobs = g * fc_flat * kT_flat                          # (n_pix,)
    live = Tobs > 0.0
    if not live.any():
        return (N.value * np.zeros(NEX))[None, :]

    Tobs_l = Tobs[live].astype(dtype, copy=False)
    fc_l = fc_flat[live]
    dO_l = dO_flat[live]
    pixel_w = (dO_l / (fc_l ** 4)).astype(dtype, copy=False)   # (n_live,)

    # ---- 4. Coarse-grid pixel sum via the fused integrand ----
    # integrand(k, p) = norm · E_obs²(k) · dΩ(p) / [ f⁴(p) · expm1(E_obs(k)/Tobs(p)) ]
    # We chunk to bound peak memory; a single broadcast at 200×200 with
    # NEC=300 is ~57 MB float32 / 114 MB float64.
    Ec = _E_mid_c.astype(dtype, copy=False)
    Ec_sq = _E_mid_c_sq.astype(dtype, copy=False)
    norm = dtype(15.0 / np.pi ** 4)
    n_live = pixel_w.size

    # Heuristic chunk size: keep the (m_E, n_live) broadcast under 32 MB.
    bytes_per_elem = np.dtype(dtype).itemsize
    chunk = max(1, (32 * 1024 * 1024) // (n_live * bytes_per_elem))
    dN_dE_c = np.zeros(NEC, dtype=np.float64)

    for k0 in range(0, NEC, chunk):
        k1 = min(k0 + chunk, NEC)
        x = Ec[k0:k1, None] / Tobs_l[None, :]          # (m_E, n_live)
        np.clip(x, 1e-30, 700.0, out=x)
        contrib = (norm * Ec_sq[k0:k1, None] / np.expm1(x)) * pixel_w[None, :]
        dN_dE_c[k0:k1] = contrib.sum(axis=1)

    # ---- 5. Linear coarse → fine interpolation, multiply by fine-bin dE ----
    dN_dE = np.interp(_E_mid, _E_mid_c, dN_dE_c, left=0.0, right=0.0)
    photar = dN_dE * _dE
    return (N.value * photar)[None, :]


# ---------------------------------------------------------------------------
# Direct-vs entry point: skip the inherited 2048-bin grid entirely.
# ---------------------------------------------------------------------------
# keV ↔ Hz conversion (= h^-1 with E in keV, ν in Hz).  Local definition so
# this module stays standalone — fitted.constants has the same number.
_KEV_TO_HZ = 2.41798924e+17


def numerical_disc_dNdE_at_vs(bh_a, rout, incl, kTdisc_array, rdisc_array,
                              vs_obs_Hz, nro=None, nphi=None, r_obs=1.0e4,
                              dtype=np.float64, source_type='keplerian',
                              colour_correction=None):
    """Observer-frame disc dN/dE evaluated **only** at user frequencies.

    Drops the inherited 2048-bin energy-grid scaffolding from
    ``numerical_disc_model`` — for a typical UV TDE light curve with
    ~10 photometric bands, the inner Planck loop runs at
    ``Nv × n_pix`` instead of ``300 × n_pix`` (a ~30× reduction in
    ``expm1`` evaluations).  Time-batched: a single ``(N_t, N_r)``
    temperature stack pays the ray-trace + mask scaffolding once
    and emits ``(N_t, N_v) dN/dE``.

    Parameters
    ----------
    bh_a : float
        BH spin parameter (M = 1 units).
    rout : float
        Outer ray-tracing boundary in r_g.
    incl : float
        Disc–observer inclination in **degrees**.
    kTdisc_array : (Nr,) or (Nt, Nr) array_like
        Per-radius rest-frame effective temperature in **keV**.  Pass
        a 2-D array to evaluate Nt epochs in one call (sharing one
        cached trace + image-plane scaffolding).
    rdisc_array : (Nr,) array_like
        Matching radii in r_g.
    vs_obs_Hz : (Nv,) array_like
        Observer-frame frequencies in Hz.  Internally converted to
        observer-frame energies in keV via ``E = ν / 2.41799e17``.
    nro, nphi : int or None
        Image-plane (radial, azimuthal) pixel counts.  ``None`` →
        ``_auto_grid(bh_a, incl)``.
    r_obs : float
        Asymptotic observer radius in r_g.  Default 1e4.
    dtype : np.dtype
        Precision of the inner Planck broadcast.  Default ``np.float64``.

    Returns
    -------
    dN_dE : ndarray, shape (Nv,) for 1-D input or (Nt, Nv) for 2-D input
        Observer-frame photon-number spectral density at the requested
        frequencies, in the same units as
        ``numerical_disc_model``'s ``dN/dE`` (i.e. ``photar / dE``
        scaled by ``N.value`` for compatibility with the FitTeD
        ``νL_ν = 4 · M_BH² · E² · dN/dE · keV_to_erg`` formula).
    """
    if numerical_disc_avaliable != "Yes":
        raise NotImplementedError(
            "the vendored ray tracer (fitted.models.raytrace) is not "
            "importable; reinstall FitTeD."
        )

    bh_a = _floor_spin(bh_a)
    incl_deg = float(incl)
    incl_rad = np.deg2rad(incl_deg)

    if nro is None or nphi is None:
        nro_auto, nphi_auto = _auto_grid(bh_a, incl_deg)
        if nro is None: nro = nro_auto
        if nphi is None: nphi = nphi_auto

    kT = np.asarray(kTdisc_array, dtype=np.float64)
    rd = np.asarray(rdisc_array, dtype=np.float64)
    squeeze = (kT.ndim == 1)
    if squeeze:
        kT = kT[None, :]
    Nt, Nr = kT.shape
    if rd.shape != (Nr,):
        raise ValueError(f"rdisc_array shape {rd.shape} doesn't match "
                         f"trailing axis of kTdisc_array ({Nr},)")

    E_obs_keV = np.asarray(vs_obs_Hz, dtype=np.float64) / _KEV_TO_HZ
    Nv = E_obs_keV.size
    out = np.zeros((Nt, Nv), dtype=np.float64)

    # ---- Trace cache (geodesic-only quantities) ----
    r_em, g, dO_flat, _valid = _cached_trace(
        bh_a, incl_rad, float(rout), int(nro), int(nphi), float(r_obs),
        source_type=source_type,
    )
    if g.size == 0:
        result = N.value * out
        return result[0] if squeeze else result

    Ec = E_obs_keV.astype(dtype, copy=False)
    Ec_sq = (Ec * Ec).astype(dtype, copy=False)
    norm = dtype(15.0 / np.pi ** 4)

    # Pre-compute the (r_em → rdisc) interpolation indices & weights once,
    # since rdisc_array is shared across all Nt epochs in this call.
    # For each pixel p we have r_em[p]; find idx[p] s.t. rd[idx-1] ≤ r_em < rd[idx].
    idx_hi = np.searchsorted(rd, r_em, side='right')
    in_range = (idx_hi > 0) & (idx_hi < Nr)
    idx_hi_c = np.clip(idx_hi, 1, Nr - 1)
    idx_lo_c = idx_hi_c - 1
    r_lo = rd[idx_lo_c]
    r_hi = rd[idx_hi_c]
    denom = r_hi - r_lo
    # Guard against repeated radii in rdisc_array (would give div-by-zero).
    denom = np.where(denom > 0.0, denom, 1.0)
    w_hi = ((r_em - r_lo) / denom) * in_range  # zero out extrapolated pixels
    w_lo = (1.0 - w_hi) * in_range

    for t in range(Nt):
        kT_t = kT[t]
        kT_flat = kT_t[idx_lo_c] * w_lo + kT_t[idx_hi_c] * w_hi
        kT_flat = np.where(kT_flat > 0.0, kT_flat, 0.0)
        # Apply user f_col(T_kelvin, r_rg) if given; else legacy Done+ 2012.
        fc_flat = _apply_user_colour_correction(colour_correction, kT_flat, r_em)

        Tobs = g * fc_flat * kT_flat
        live = Tobs > 0.0
        if not live.any():
            continue
        Tobs_l = Tobs[live].astype(dtype, copy=False)
        fc_l = fc_flat[live]
        dO_l = dO_flat[live]
        pixel_w = (dO_l / (fc_l ** 4)).astype(dtype, copy=False)

        # Single broadcast (Nv, n_live).  For typical TDE workloads
        # Nv ~ 10–100 and n_live ~ 5e3–6e4, so ~MB-scale — no chunking.
        n_live = pixel_w.size
        bytes_per_elem = np.dtype(dtype).itemsize
        max_bytes = 64 * 1024 * 1024
        chunk = max(1, max_bytes // max(1, n_live * bytes_per_elem))
        if chunk >= Nv:
            x = Ec[:, None] / Tobs_l[None, :]
            np.clip(x, 1e-30, 700.0, out=x)
            contrib = (norm * Ec_sq[:, None] / np.expm1(x)) * pixel_w[None, :]
            out[t] = contrib.sum(axis=1)
        else:
            for k0 in range(0, Nv, chunk):
                k1 = min(k0 + chunk, Nv)
                x = Ec[k0:k1, None] / Tobs_l[None, :]
                np.clip(x, 1e-30, 700.0, out=x)
                contrib = (norm * Ec_sq[k0:k1, None] / np.expm1(x)) * pixel_w[None, :]
                out[t, k0:k1] = contrib.sum(axis=1)

    out *= N.value
    return out[0] if squeeze else out
