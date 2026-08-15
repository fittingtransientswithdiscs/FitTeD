"""Geometry-only orchestrator for intensity (thermal / unpolarized) tracing.

Returns the minimal geometric data needed to weight a thermal (or any
isotropic / direction-resolved) emissivity onto the observer screen:

    (r_em, theta_em, g, valid)

with g = ω_obs / ω_em the photon redshift factor between an emitter at
(r_em, θ_em) and an asymptotic static observer.

This is the "intensity track" of polarized_trace — same geodesic batch,
same redshift utility — but with the polarization machinery (emission
model, Walker–Penrose, screen tetrad, Stokes projection) skipped, **and**
also with the (t, φ) closed-form Mino-time machinery skipped: an
asymptotic SED never reads coordinate t or φ at the hit, only the
geometric quantities (r_em, θ_em) and the photon momentum sign data
needed to evaluate g.  We therefore call ``first_hit_batch`` with
``include_tphi=False``, which avoids the per-ray elliptic-integral
work in ``_batch_tphi.radial_J_batch`` / ``_I1_scat_batch`` and the
polar G accumulators in ``geodesic_batch.hit_event``.

Concretely, the cold trace at 200×200 pixels drops from ~240 ms to
~33 ms (≳7× faster) on a typical (a, incl) without changing g or
r_em to machine precision (verified bit-identical against the
``include_tphi=True`` path).

This is what FitTeD's gr_disc_plus_new wrapper needs to weight a
Novikov–Thorne T(r) profile through the Planck specific intensity
I_ν ∝ B_ν(T) and form an observed SED via

    F_E ∝ Σ_pixels  g³ · B_{E_obs / g}(f_col · T_eff(r_em)) / f_col⁴ · dΩ_pix.

The (r_em, g, valid) outputs are exactly the ones the Fortran
`tdedisc_grid` already returns to numerical_disc_model — keeping the
swap drop-in.

The function is exported from the package under two names:

    kerrgeo.intensity_trace   — historical name; still the SED entry-point.
    kerrgeo.geometry_trace    — alias making the geometry-only contract
                                explicit at the call site.

Both bind to the same implementation; pick whichever reads better in
the calling code.

Notes
-----
* No emission-angle (limb-darkening) factor is computed; for a thermal
  disk the BH→pixel limb darkening is folded into the bolometric flux
  through the per-pixel geometric weight (∼ cos i) of the asymptotic
  camera, not into the local source function.
* For an emitter inside the ISCO `Source.u_of` returns NaN (per the
  KeplerianDisk contract); those pixels appear as `valid=False` with
  NaN in the other outputs.
* The (t, φ)-skip relies on the target's ``first_hit_batch`` being
  willing to honour ``include_tphi=False``.  ``EquatorialDisk`` does;
  in addition, when paired with an ``AsymptoticObserver`` it also
  surfaces a closed-form ``s_theta_at_hit = +1`` (correct for any
  first equator crossing of the polar oscillation), letting
  ``p_at_hit`` skip even the polar-anchor / ``ellipj`` recovery step.
"""

import numpy as np

from .redshift import redshift_factor, static_observer_4velocity


def intensity_trace(a, observer, target, source, pixels):
    """Trace observer→target and compute the per-pixel redshift factor.

    Parameters
    ----------
    a : float
        BH spin (M = 1).
    observer : kerrgeo.observers.Observer
        Source of the photon batch (e.g. AsymptoticObserver).
    target : kerrgeo.targets.Target
        First-hit hypersurface (e.g. EquatorialDisk).
    source : kerrgeo.sources.Source
        Provides the emitter 4-velocity ``u_of(r, θ)``; only the
        kinematics matter — no emission model is consulted.  For a
        thin disk pass ``KeplerianDisk(a, prograde=True)``.
    pixels : dict
        Observer-specific (for AsymptoticObserver, ``dict(alpha=, beta=)``).

    Returns
    -------
    out : dict
        ``r_hit``      — (..., ) emission-radius array, NaN where the
                         photon misses the target.
        ``theta_hit``  — (..., ) emission-θ array (= π/2 for an
                         equatorial disk where the ray hits).
        ``g``          — (..., ) redshift factor ω_obs / ω_em.  NaN
                         on missed / inside-ISCO rays.
        ``valid``      — (..., ) boolean: True for rays that landed
                         on the target with a well-defined u_em.
        ``shape``      — original broadcast shape of ``pixels``.

    The (r_hit, theta_hit, g) arrays are reshaped to the input pixel
    shape so callers can drop them straight into a 2-d image grid.
    """
    # ---- 1. Geodesic batch and first-hit event ------------------------
    # Geometry-only path (include_tphi=False): we never read t_hit, phi_hit,
    # tau, sigma, G_tau/sig or J_tau/sig from `event` below — only r_hit,
    # theta_hit, s_r_at_hit (and indirectly s_theta_at_hit, recovered from
    # lam_hit by `batch.p_at_hit`).  Skipping include_tphi=True shaves the
    # full radial+polar (t, φ) elliptic-integral machinery (radial_J_batch,
    # polar G batch in hit_event, _I1_scat_batch) out of the critical path.
    batch, ray_meta, shape = observer.make_batch(a, pixels)
    event = target.first_hit_batch(batch, ray_metadata=ray_meta,
                                    include_tphi=False)

    r_em  = np.asarray(event['r_hit'])
    th_em = np.asarray(event['theta_hit'])
    hit = np.isfinite(r_em) & np.isfinite(th_em)

    # ---- 2. Emitter / observer 4-velocities and 4-momenta -------------
    u_em = source.u_of(r_em, th_em)
    valid = hit & np.isfinite(u_em[..., 0])

    r_safe  = np.where(hit, r_em,  10.0)
    th_safe = np.where(hit, th_em, 0.5 * np.pi)
    p_em = batch.p_at_hit(event)

    r_obs  = np.full(batch.N, observer.r_obs)
    th_obs = np.full(batch.N, observer.inclination)
    p_obs = batch.p_at_observer()
    u_obs = static_observer_4velocity(a, r_obs, th_obs)

    # ---- 3. Redshift factor ------------------------------------------
    g = redshift_factor(a, u_em, p_em, r_safe, th_safe,
                            u_obs, p_obs, r_obs, th_obs)

    # ---- 4. Mask off invalid pixels ----------------------------------
    def _mask(arr):
        return np.where(valid, arr, np.nan)

    return dict(
        r_hit     = np.where(hit, r_em, np.nan).reshape(shape),
        theta_hit = np.where(hit, th_em, np.nan).reshape(shape),
        g         = _mask(g).reshape(shape),
        valid     = valid.reshape(shape),
        shape     = shape,
    )


# Public alias.  The body above is geometry-only (it never reads coordinate
# t or φ at the hit), so ``geometry_trace`` is the more descriptive name for
# new call sites; ``intensity_trace`` is the historical name kept for
# backward compatibility with FitTeD's gr_disc_plus_new wrapper.
geometry_trace = intensity_trace
