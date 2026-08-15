"""Closed-form camera-at-infinity ray-tracer for Kerr equatorial disks.

For each image-plane pixel (α, β) at observer inclination i:
  1.  Bardeen transform → (ξ, η, μ_obs, s_μ_obs)
  2.  Polar Mino-time from observer down to μ=0  →  λ_polar
  3.  Radial inversion r(λ_polar) from observer anchored at r=∞

Output: R_hit(α, β) — the BL radius at which the backward ray first
crosses the equatorial plane (or NaN if the pixel is "shadow", i.e.
the trajectory reaches the horizon or wraps past the past-infinity
turning point before the polar motion brings μ to zero).

Higher-order images (n ≥ 1) are accessed via the `n` parameter of
`camera_hit_radius` (which adds n·Γ_p to λ_polar before the radial
inversion), or via `camera_hit_radius_multi` which returns the full
per-pixel sequence (r_hit_n, code_n, s_r_disk_n, lam_hit_n) for
n = 0, 1, ..., n_max in one call.

The radial piece reuses the outer-anchor Weierstrass machinery from
return_pipeline_fast.py.  The polar piece uses Legendre F(φ, k).

Time t and azimuth φ are deferred — n = 0 radius map only.
"""

import numpy as np
from scipy.special import ellipk, ellipkinc, ellipj, elliprf

from .return_pipeline_fast import (
    _cardano_three_real,
    _cardano_one_real,
    _quartic_largest_real,
    _wp_inv_uniform,
    _wp_forward_sn,
    _wp_forward_cn,
)
from .bardeen import bardeen_transform, S_MU_SIGN

# Fate codes for pixels
PIX_HIT = 0       # ray crossed equator at finite r > r_+
PIX_SHADOW = 1    # ray plunged to horizon before crossing equator
PIX_OUT = 2       # ray bounced at R_out and went to past infinity without crossing equator
PIX_FORBIDDEN = 3 # observer's μ lies outside the allowed polar libration range


# =====================================================================
# Polar Mino time from observer to equator (for n = 0 image)
# =====================================================================
def polar_mino_to_equator(a, xi, eta, mu_obs, s_mu_obs):
    """Compute λ_polar: Mino time for the direct (n=0) backward photon
    to go from μ = μ_obs at the observer down to μ = 0 on the equator.

    Also returns the polar half-period Γ_p (for n ≥ 1 stepping) and
    an 'allowed' mask (False means μ_obs is outside the libration band).

    All inputs are numpy arrays of common broadcast shape.
    """
    a = float(a)
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    mu_obs = np.asarray(mu_obs, dtype=np.float64)
    s_mu_obs = np.asarray(s_mu_obs, dtype=np.float64)

    # Polar turning-point quadratic in v = a²μ² (same as pipeline):
    #   v² + Bc·v + Cc = 0,   Bc = η+ξ²-a²,   Cc = -a²η
    Bc = eta + xi * xi - a * a
    disc = Bc * Bc + 4.0 * a * a * eta
    ok_disc = disc >= 0.0
    sd = np.sqrt(np.where(ok_disc, disc, 0.0))

    # Numerically stable up (the larger root; upper μ² turning point):
    #   Bc ≥ 0:  up = 2 a² η / (sd + Bc)     (cancellation-free)
    #   Bc < 0:  up = 0.5·(-Bc + sd)         (no cancellation anyway)
    up = np.where(Bc >= 0.0,
                  2.0 * a * a * eta / np.where(sd + Bc > 1e-300, sd + Bc, 1.0),
                  0.5 * (-Bc + sd))
    um = -0.5 * (Bc + sd)         # the smaller root (typically < 0)
    udiff = up - um                # = sd, positive

    # up_u = up / a², the μ² upper turning point.  Cancellation-free form
    # equals 2η/(sd+Bc) when Bc ≥ 0 — safe down to a = 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        up_u = np.where(Bc >= 0.0,
                        2.0 * eta / np.where(sd + Bc > 1e-300, sd + Bc, 1.0),
                        up / np.maximum(a * a, 1e-300))
    up_u = np.where(a == 0.0,
                    np.where(eta + xi * xi > 0.0,
                             eta / np.maximum(eta + xi * xi, 1e-300),
                             0.0),
                    up_u)

    # Allowed region: μ_obs² ≤ up_u   (observer inside the polar libration band)
    allowed = ok_disc & (up_u > 0.0) & (mu_obs * mu_obs <= up_u + 1e-12) \
              & (udiff > 0.0)

    # Safe guards for unallowed pixels
    up_u_safe = np.where(allowed, up_u, 1.0)
    udiff_safe = np.where(allowed, udiff, 1.0)
    up_safe = np.where(allowed, up, 1.0)

    # cos ψ_obs = μ_obs / √up_u     (stable at a=0 since up_u is finite)
    cos_psi = np.clip(mu_obs / np.sqrt(up_u_safe), -1.0, 1.0)
    psi_obs = np.arccos(cos_psi)

    # Elliptic modulus: k² = up / (up-um) = a²·up_u / udiff; 0 at a=0
    k2 = np.where(allowed, a * a * up_u_safe / udiff_safe, 0.0)
    k2 = np.clip(k2, 0.0, 1.0 - 1e-15)

    K_full = ellipk(k2)
    F_psi = ellipkinc(psi_obs, k2)

    inv_sqrt_udiff = 1.0 / np.sqrt(udiff_safe)
    Gp = 2.0 * K_full * inv_sqrt_udiff
    T_partial = (K_full - F_psi) * inv_sqrt_udiff

    # For s_μ_forward_obs = +1  (forward photon has dμ/dλ > 0 at observer;
    #   backward ray's μ decreases monotonically to 0):
    #        λ_polar_0 = T_partial
    # For s_μ_forward_obs = −1  (backward ray must first rise to μ_max
    #   then descend to equator):
    #        λ_polar_0 = Γ_p − T_partial
    direct = s_mu_obs > 0.0
    lam_polar = np.where(direct, T_partial, Gp - T_partial)

    # Mark disallowed pixels
    lam_polar = np.where(allowed, lam_polar, np.nan)
    Gp = np.where(allowed, Gp, np.nan)
    return lam_polar, Gp, allowed


# =====================================================================
# Radial inversion from infinity
# =====================================================================
def radial_from_infinity(a, xi, eta, lam_polar):
    """Given (ξ, η) and a backward Mino time λ_polar, return the BL radius
    at which the photon (coming from r = ∞, backward-traced) reaches that
    Mino time.

    Shadow conditions:
      MONO (rp > R_out or rect regime):
          λ_polar > (mu_esc - mu_rp)  →  photon past the horizon
      OUTER (4 real roots, R_out > rp):
          λ_polar > 2·mu_esc         →  photon past past-infinity

    Returns (r_hit, zone_code) where
        zone_code ∈ {PIX_HIT, PIX_SHADOW, PIX_OUT}.
    r_hit is NaN on non-HIT pixels.
    """
    a = float(a)
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    lam_polar = np.asarray(lam_polar, dtype=np.float64)
    shape = np.broadcast_shapes(xi.shape, eta.shape, lam_polar.shape)
    xi = np.broadcast_to(xi, shape).copy()
    eta = np.broadcast_to(eta, shape).copy()
    lam_polar = np.broadcast_to(lam_polar, shape).copy()

    # PD variables + quartic coefficients
    xi_PD = a * (a - xi)
    eta_PD = eta + (xi - a) ** 2
    b = 2.0 * xi_PD - eta_PD
    c = 2.0 * eta_PD
    d = xi_PD ** 2 - eta_PD * a * a

    # Cieślik invariants
    a2 = (2.0 * xi_PD - eta_PD) / 6.0
    a3 = eta_PD / 2.0
    a4 = xi_PD ** 2 - eta_PD * a * a
    g2 = a4 + 3.0 * a2 * a2
    g3 = a2 * a4 - a2 ** 3 - a3 * a3
    Delta_W = g2 ** 3 - 27.0 * g3 ** 2

    # Horizon
    rp = 1.0 + np.sqrt(max(1.0 - a * a, 0.0))

    is_scat = Delta_W > 0.0
    is_rect = Delta_W < 0.0

    r_hit = np.full(shape, np.nan)
    zone_code = np.full(shape, PIX_SHADOW, dtype=np.int8)
    # s_r_disk = sign of dr/dτ of the FORWARD photon at the disk:
    #   +1 = photon emitted outgoing (no radial turning between disk and observer)
    #   -1 = photon emitted ingoing  (photon bounced off r_min between disk and observer)
    # Meaningful only on HIT pixels; defaults to +1 elsewhere.
    s_r_disk = np.ones(shape, dtype=np.float64)

    def _process(mask, e_triple_fn, wp_forward_fn):
        if not mask.any():
            return
        g2m = g2[mask]; g3m = g3[mask]
        b_m = b[mask]; c_m = c[mask]; d_m = d[mask]
        lam_m = lam_polar[mask]

        # Largest real root R_out of ψ_q
        R_out = _quartic_largest_real(b_m, c_m, d_m)
        valid_R = np.isfinite(R_out)

        # Biermann outer-anchor quantities
        phi_R = (12.0 * R_out * R_out + 2.0 * b_m) / 24.0
        psi_p = 4.0 * R_out ** 3 + 2.0 * b_m * R_out + c_m

        def W(r):
            # FitTeD vendored copy: r == R_out, and non-finite R_out, divide by
            # zero here.  Those pixels are discarded downstream by valid_R /
            # in_range, so the non-finite result is intentional; errstate
            # silences the warning without changing any value.
            with np.errstate(invalid='ignore', divide='ignore'):
                return phi_R + psi_p / (4.0 * (r - R_out))

        # e-triple
        e1, e2, e3 = e_triple_fn(g2m, g3m)

        # Inverse ℘ anchors
        W_inf = phi_R
        mu_esc = _wp_inv_uniform(W_inf, e1, e2, e3)
        W_rp = W(rp)
        mu_rp = _wp_inv_uniform(W_rp, e1, e2, e3)

        horizon_above = rp > R_out

        # Two zones:
        # OUTER: horizon_above = False.  Photon from ∞ bounces at R_out
        #        and returns to ∞ in total Mino time 2·mu_esc.
        # MONO:  horizon_above = True.   Photon from ∞ plunges to horizon
        #        in total Mino time (mu_esc − mu_rp).
        tmax_outer = 2.0 * mu_esc
        tmax_mono = mu_esc - mu_rp
        tmax = np.where(horizon_above, tmax_mono, tmax_outer)

        in_range = (lam_m >= 0.0) & (lam_m <= tmax) & valid_R

        # Evaluate r(λ_anchor = mu_esc − λ_polar)
        # ℘ is even, so we can just pass |mu_esc − lam_polar|.
        lam_anchor = mu_esc - lam_m
        wp_val = wp_forward_fn(np.abs(lam_anchor), e1, e2, e3)
        # See the note on W() above -- masked downstream by in_range.
        with np.errstate(invalid='ignore', divide='ignore'):
            r_val = R_out + psi_p / (4.0 * (wp_val - phi_R))

        # Sub-classify
        sub_code = np.where(in_range, PIX_HIT,
                            np.where(horizon_above, PIX_SHADOW, PIX_OUT))
        sub_code = np.where(valid_R, sub_code, PIX_SHADOW).astype(np.int8)
        sub_r = np.where(in_range, r_val, np.nan)

        # `s_r_disk` is the direct/bounce flag: +1 = the BACKWARD-trace
        # photon (observer → disc) descends monotonically with no
        # radial turning point on the way to the disc; −1 = it bounces
        # off pericenter R_out > r_+ before hitting the disc (only
        # possible in the OUTER regime).
        #
        # In MONO regime (horizon_above = True) no bounce can happen —
        # the photon plunges all the way from r_obs to r_+, so the disc
        # crossing is always direct: s_r_disk = +1.  We force this
        # explicitly because the W₂ inverse Weierstrass anchor (mu_rp)
        # can land on a non-principal branch for MONO+rect pixels,
        # giving lam_anchor a spurious sign that has no physical
        # bouncing meaning.
        #
        # In OUTER regime, lam_anchor = mu_esc - lam_polar crosses 0
        # at the pericenter; sign(lam_anchor) is the correct
        # direct/bounce flag there.
        sign_la_outer = np.where(lam_anchor >= 0.0, 1.0, -1.0)
        direct_or_bounce = np.where(horizon_above, 1.0, sign_la_outer)
        sub_sr = np.where(in_range, direct_or_bounce, 1.0)

        zone_code[mask] = sub_code
        r_hit[mask] = sub_r
        s_r_disk[mask] = sub_sr

    # -- scat branch -----------------------------------------------------
    def e_triple_scat(g2m, g3m):
        e3s, e2s, e1s = _cardano_three_real(-g2m / 4.0, -g3m / 4.0)
        return e1s, e2s, e3s

    def wp_fwd_scat(lam, e1, e2, e3):
        return _wp_forward_sn(lam, e1, e2, e3)

    # -- rect branch -----------------------------------------------------
    def e_triple_rect(g2m, g3m):
        e_r = _cardano_one_real(-g2m / 4.0, -g3m / 4.0)
        rad = np.sqrt(np.maximum(3.0 * e_r * e_r - g2m, 0.0)) / 2.0
        e_c1 = (-e_r / 2.0 + 1j * rad)
        e_c2 = e_c1.conjugate()
        return e_r, e_c1, e_c2

    def wp_fwd_rect(lam, e_r, e_c1, e_c2):
        H = np.sqrt((e_r - np.real(e_c1)) ** 2 + np.imag(e_c1) ** 2)
        kp2 = 0.5 - 3.0 * e_r / (4.0 * H)
        return _wp_forward_cn(lam, e_r, H, kp2)

    _process(is_scat, e_triple_scat, wp_fwd_scat)
    _process(is_rect, e_triple_rect, wp_fwd_rect)
    return r_hit, zone_code, s_r_disk


# =====================================================================
# Multi-image generalisation: per-pixel sequences over n
# =====================================================================
def camera_hit_radius_multi(a, i, alpha, beta, n_max=3):
    """Per-pixel sequences of (r_hit, code, s_r_disk, lam_hit) for n = 0..n_max.

    For an observer at r = ∞, inclination i, this back-traces each
    image-plane pixel and records the BL radius of the n-th equator
    crossing for n = 0, 1, ..., n_max.  n=0 reproduces
    ``camera_hit_radius(a, i, alpha, beta, n=0)`` bit-for-bit;
    higher n's correspond to photons that wind around the photon
    sphere n times before reaching the observer.

    The polar anchor (which gives Γ_p) and the radial anchor
    (Weierstrass invariants, R_out, mu_esc, mu_rp) are computed once
    per pixel; the loop over n only re-evaluates the forward ℘ at
    λ_polar_0 + n·Γ_p.

    Plunging-region landings (r_+ < r_hit < r_ISCO) are returned with
    `code == PIX_HIT` just like Keplerian-region landings — the
    function knows nothing about ISCO.  Downstream consumers should
    consult `KerrDisk.u_of(r_hit, π/2)` (which itself dispatches
    Keplerian vs Thorne-plunging on whether r_hit ≷ r_ISCO) to get
    the correct rest-frame u^μ for the g factor, and use the
    returned `s_r_disk` for the k_r·u^r contribution that's
    nonzero in the plunging branch.

    Parameters
    ----------
    a, i : float
        Kerr spin and observer inclination (radians).
    alpha, beta : array_like
        Image-plane coordinates, common broadcast shape ``pix.shape``.
    n_max : int, default 3
        Highest image order to track.  Output arrays have shape
        ``(n_max + 1,) + pix.shape``, with the leading axis indexing n.

    Returns
    -------
    dict with keys
        r_hit    : (n_max+1, *pix.shape) float
            BL radius of the n-th equator crossing.  NaN on non-HIT
            pixels (PIX_SHADOW / PIX_OUT / PIX_FORBIDDEN).
        code     : (n_max+1, *pix.shape) int8
            Per-(n, pixel) fate code.  Once a pixel becomes non-HIT
            at some n it usually stays non-HIT at higher n's
            (the photon has plunged or escaped past past-infinity),
            but the test is per-n so this is not strictly monotonic.
        s_r_disk : (n_max+1, *pix.shape) float
            Sign of dr/dλ of the FORWARD photon at the n-th crossing
            (+1 direct / outgoing leg, −1 bounced off pericenter).
            Set to +1 on non-HIT pixels by convention.
        lam_hit  : (n_max+1, *pix.shape) float
            Forward Mino time at the n-th crossing (= λ_polar_0 + n·Γ_p
            on HIT pixels, NaN otherwise).
        xi, eta, mu_obs : (*pix.shape,) float
            Per-pixel conserved quantities and observer cos θ_obs.

    Notes
    -----
    For the mirror multi-bounce observed-spectrum sum at infinity,
    each (n, pixel) contributes an emitter at (r_em, ξ) = (r_hit[n,p],
    xi[p]) with redshift

        g_n(p) = 1 / (u^t(r_em) − ξ·u^φ(r_em) − k_r·u^r(r_em))

    where u^μ is the disc-fluid 4-velocity at r_em (Keplerian outside
    ISCO; Thorne plunging continuation inside) and k_r at the crossing
    is s_r_disk[n,p] · √R(r_em) / Δ.  The intensity arriving at the
    observer pixel from this emitter is then g_n³ · I_em(ν / g_n,
    r_em), summed over n.
    """
    n_max = int(n_max)
    xi, eta, mu_obs, s_mu_obs = bardeen_transform(a, i, alpha, beta)
    lam_polar_0, Gp, allowed = polar_mino_to_equator(
        a, xi, eta, mu_obs, s_mu_obs,
    )

    pix_shape = np.shape(np.broadcast_to(np.asarray(alpha), np.shape(beta)))
    out_shape = (n_max + 1,) + pix_shape

    r_hit    = np.full(out_shape, np.nan)
    code     = np.full(out_shape, PIX_SHADOW, dtype=np.int8)
    s_r_disk = np.ones(out_shape)
    lam_hit  = np.full(out_shape, np.nan)

    for n in range(n_max + 1):
        lam_polar_n = lam_polar_0 + n * Gp
        r_n, zone_n, sr_n = radial_from_infinity(a, xi, eta, lam_polar_n)
        zone_n = np.where(allowed, zone_n, PIX_FORBIDDEN).astype(np.int8)
        is_hit = (zone_n == PIX_HIT)

        r_hit[n]    = np.where(is_hit, r_n, np.nan)
        code[n]     = zone_n
        s_r_disk[n] = np.where(is_hit, sr_n, 1.0)
        lam_hit[n]  = np.where(is_hit, lam_polar_n, np.nan)

    return dict(
        r_hit=r_hit, code=code,
        s_r_disk=s_r_disk, lam_hit=lam_hit,
        xi=xi, eta=eta, mu_obs=mu_obs,
    )


# =====================================================================
# Public API: re-export camera_hit_event from camera_tphi.
# Users that just want (r_hit, t_hit, φ_hit, s_hit) for a batch of pixels
# can import both camera_hit_radius and camera_hit_event from this module.
# A lazy import avoids a circular dependency (camera_tphi imports
# camera_hit_radius from this file).
# =====================================================================
def camera_hit_event(a, i, alpha, beta, regularize=True, r_obs=None):
    """Thin re-export of `camera_tphi.camera_hit_event` — computes the full
    equatorial-hit event (r, t, φ, s) for each image-plane pixel.  See
    `camera_tphi.camera_hit_event` for the full docstring.
    """
    from .camera_tphi import camera_hit_event as _chit
    return _chit(a, i, alpha, beta, regularize=regularize, r_obs=r_obs)


# =====================================================================
# End-to-end camera ray-tracer
# =====================================================================
def camera_hit_radius(a, i, alpha, beta, n=0, return_sr=False,
                      return_lam=False):
    """Map image-plane pixels (α, β) to the BL radius where the backward
    ray first meets the equatorial plane, for camera at r = ∞, inclination i.

    Parameters
    ----------
    a      : float
        Kerr spin.
    i      : float
        Observer inclination (radians).
    alpha, beta : array_like
        Image-plane coordinates, common broadcast shape.
    n      : int
        Image order (0 = direct).  For n ≥ 1, adds n·Γ_p to λ_polar.
    return_sr : bool
        If True, also return the forward-photon radial sign s_r at the disk
        (+1 outgoing, −1 ingoing / bounced).
    return_lam : bool
        If True, also return ``lam_polar`` — the forward-photon Mino time
        at the equator crossing (NaN on non-HIT pixels).  This is exactly
        the value ``geodesic_batch.lam_of_theta(π/2, n_crossing=n)`` would
        recompute from the polar anchor; surfacing it here lets downstream
        consumers (e.g. ``EquatorialDisk.first_hit_batch`` with
        ``include_tphi=False``) recover ``s_theta_at_hit`` via a single
        cheap ``ellipj`` evaluation rather than the full elliptic-integral
        (t, φ) machinery.

    Returns
    -------
    r_hit     : float array
        Radius of first equator crossing; NaN on shadow / forbidden pixels.
    pix_code  : int8 array (PIX_HIT / PIX_SHADOW / PIX_OUT / PIX_FORBIDDEN)
    s_r_disk  : float array (only if return_sr=True)
    lam_hit   : float array (only if return_lam=True)
    """
    xi, eta, mu_obs, s_mu_obs = bardeen_transform(a, i, alpha, beta)
    lam_polar, Gp, allowed = polar_mino_to_equator(a, xi, eta, mu_obs, s_mu_obs)

    if n != 0:
        lam_polar = lam_polar + n * Gp

    r_hit, zone_code, s_r_disk = radial_from_infinity(a, xi, eta, lam_polar)

    # Forbidden (polar sector inaccessible) overrides whatever radial said
    zone_code = np.where(allowed, zone_code, PIX_FORBIDDEN).astype(np.int8)
    r_hit = np.where(zone_code == PIX_HIT, r_hit, np.nan)
    s_r_disk = np.where(zone_code == PIX_HIT, s_r_disk, 1.0)
    lam_hit = np.where(zone_code == PIX_HIT, lam_polar, np.nan)
    if return_sr and return_lam:
        return r_hit, zone_code, s_r_disk, lam_hit
    if return_sr:
        return r_hit, zone_code, s_r_disk
    if return_lam:
        return r_hit, zone_code, lam_hit
    return r_hit, zone_code


# =====================================================================
# SELF-TEST
# =====================================================================
