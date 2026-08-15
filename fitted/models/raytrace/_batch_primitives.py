"""Vectorised closed-form primitives for Phase 4b batch tracing.

This module is a batch-of-rays counterpart to the per-ray scalar
primitives in `tphi_radial` and `tphi_elliptic`.  All functions take
length-N numpy arrays (with `a` typically scalar, broadcast) and return
arrays of the same shape.

The underlying math is bit-for-bit identical to the scalar path — the
only change is that each function processes the whole batch in a
single broadcast op instead of one photon at a time.  Regime branching
(scat vs rect Weierstrass) is handled by running both branches on
masked subsets and merging.

Dependencies leveraged (all already vectorised in the legacy code):
  - `_quartic_largest_real(b, c, d)` → R_out per ray
  - `_cardano_three_real(p, q)` → (e1, e2, e3) for scat
  - `_cardano_one_real(p, q)` → e_r for rect
  - `_wp_inv_uniform(W, e1, e2, e3)` → λ_W via Carlson R_F
  - `ellipj`, `ellipkinc`, `ellipk`, `ellipe`, `ellipeinc` → NumPy ufuncs

Functions provided:
  radial_anchor_batch(a, xi, eta, r_obs)
      -> dict with 'regime_mask' ('scat'/'rect' per ray) and per-ray
         Biermann anchor arrays + Weierstrass invariants + λ_obs_W.
  wp_forward_batch(lam_W, anchor)
      -> (℘, ℘') arrays; evaluates the correct regime per ray via mask.
  wp_inv_batch(W, anchor)
      -> λ_W inverse, per ray, regime-aware.
  polar_anchor_batch(a, xi, eta, mu_obs, s_mu_fwd)
      -> dict with up, um, udiff, sqrt_udiff, mu_plus, k2, u_obs, K_comp
         — everything the polar Jacobi evaluator needs.
"""

import numpy as np
from scipy.special import ellipj, ellipk, ellipkinc

from .return_pipeline_fast import (
    _quartic_largest_real, _cardano_three_real, _cardano_one_real,
    _wp_inv_uniform,
)


# ===========================================================================
# Radial anchor — vectorised _radial_anchor
# ===========================================================================

def radial_anchor_batch(a, xi, eta, r_obs):
    """Per-ray Biermann anchor + Weierstrass invariants + λ_obs_W.

    Parameters
    ----------
    a      : scalar, Kerr spin.
    xi, eta: arrays of shape (N,), conserved quantities per ray.
    r_obs  : scalar or array broadcastable to (N,), observer radius.

    Returns
    -------
    dict with arrays shaped (N,):
        regime       : 'scat' or 'rect' per ray (as a uint8 array: 0 scat, 1 rect)
        valid        : bool, False for marginal or degenerate rays
        R_out        : Biermann anchor (largest real quartic root)
        phi_R, psi_p : Biermann anchor derivatives
        e1, e2, e3   : Weierstrass e-triple (real for scat; e_r + complex
                        conj pair for rect, packaged as (e_r, e_c1, e_c2))
        g2, g3       : Weierstrass invariants
        r_plus, r_minus : horizon radii (scalar, broadcast)
        lam_obs_W    : Biermann parameter at r = r_obs
    """
    a = float(a)
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    r_obs = np.asarray(r_obs, dtype=np.float64)
    if r_obs.ndim == 0:
        r_obs = np.broadcast_to(r_obs, xi.shape).astype(np.float64)

    a2 = a * a
    xi_PD = a * (a - xi)
    eta_PD = eta + (xi - a) ** 2

    # Quartic coefficients:  R(q) = q⁴ + b·q² + c·q + d
    b = 2.0 * xi_PD - eta_PD
    c = 2.0 * eta_PD
    d = xi_PD ** 2 - eta_PD * a2

    # Cieślik invariants
    a2_inv = (2.0 * xi_PD - eta_PD) / 6.0
    a3_inv = eta_PD / 2.0
    a4_inv = xi_PD ** 2 - eta_PD * a2
    g2 = a4_inv + 3.0 * a2_inv * a2_inv
    g3 = a2_inv * a4_inv - a2_inv ** 3 - a3_inv * a3_inv
    Delta_W = g2 ** 3 - 27.0 * g3 ** 2

    is_scat = Delta_W > 0.0
    is_rect = Delta_W < 0.0
    is_valid = (np.abs(Delta_W) >= 1e-14)

    # Biermann anchor: largest real root of the radial quartic.
    R_out = _quartic_largest_real(b, c, d)
    # Non-finite R_out ⇒ invalid ray
    valid = is_valid & np.isfinite(R_out)

    phi_R = (12.0 * R_out * R_out + 2.0 * b) / 24.0
    psi_p = 4.0 * R_out ** 3 + 2.0 * b * R_out + c

    # Horizon roots (scalar, broadcast)
    disc_hor = max(1.0 - a2, 0.0)
    sh = np.sqrt(disc_hor)
    r_plus = 1.0 + sh
    r_minus = 1.0 - sh

    # Scat regime: three real e-values (e₃ ≤ e₂ ≤ e₁).  _cardano_three_real
    # returns sorted ascending (t0 ≤ t1 ≤ t2), so e3 = t0, e2 = t1, e1 = t2.
    # (_radial_anchor does exactly the same assignment.)
    e3_s, e2_s, e1_s = _cardano_three_real(-g2 / 4.0, -g3 / 4.0)
    # Rect regime: single real e_r
    e_r = _cardano_one_real(-g2 / 4.0, -g3 / 4.0)
    # For rect, construct the complex-conjugate pair that feeds
    # _wp_inv_uniform directly.
    rad_c = np.sqrt(np.maximum(3.0 * e_r * e_r - g2, 0.0)) / 2.0
    e_c1 = -e_r / 2.0 + 1j * rad_c          # complex
    e_c2 = e_c1.conjugate()                  # complex

    # Auxiliary rect params for forward ℘ (cn-based)
    H = np.sqrt(np.maximum(3.0 * e_r * e_r - g2 / 4.0, 0.0))
    # kp2 is only well-defined when H > 0; leave NaN elsewhere and rely on
    # regime mask.
    with np.errstate(invalid='ignore', divide='ignore'):
        kp2 = 0.5 - 3.0 * e_r / (4.0 * H)

    # W(r) at observer, per ray
    with np.errstate(invalid='ignore', divide='ignore'):
        W_obs = phi_R + psi_p / (4.0 * (r_obs - R_out))

    # λ_obs_W via Carlson R_F — regime-masked to use real triple for scat,
    # (e_r, e_c1, e_c2) for rect.  Run each on its own mask; merge.
    lam_obs_W = np.full_like(xi, np.nan, dtype=np.float64)

    if np.any(is_scat):
        m = is_scat & valid
        if np.any(m):
            lam_obs_W[m] = np.real(_wp_inv_uniform(
                W_obs[m], e1_s[m], e2_s[m], e3_s[m]))
    if np.any(is_rect):
        m = is_rect & valid
        if np.any(m):
            lam_obs_W[m] = np.real(_wp_inv_uniform(
                W_obs[m], e_r[m], e_c1[m], e_c2[m]))

    # 0 = scat, 1 = rect
    regime = np.where(is_scat, 0, np.where(is_rect, 1, -1)).astype(np.int8)

    return dict(
        regime=regime,
        valid=valid,
        R_out=R_out,
        phi_R=phi_R,
        psi_p=psi_p,
        e1=e1_s, e2=e2_s, e3=e3_s,
        e_r=e_r, e_c1=e_c1, e_c2=e_c2,
        H=H, kp2=kp2,
        g2=g2, g3=g3,
        r_plus=r_plus, r_minus=r_minus,
        lam_obs_W=lam_obs_W,
        W_obs=W_obs,
        r_obs=r_obs,
        # Phase 4c extras: carried through so radial_J_batch can
        # reconstruct xi_PD and a² without re-parsing conserved quantities.
        a_scalar=a,
        xi_arr=xi,
        eta_arr=eta,
    )


# ===========================================================================
# Forward ℘(λ_W), ℘'(λ_W) — regime-aware batch evaluator
# ===========================================================================

def wp_forward_batch(lam_W, anchor):
    """Evaluate (℘(λ_W), ℘'(λ_W)) per ray, regime-masked.

    Parameters
    ----------
    lam_W : (N,) array — Biermann parameter per ray.
    anchor: dict from `radial_anchor_batch`.

    Returns
    -------
    wp, wpp : (N,) arrays, NaN on invalid rays.
    """
    lam_W = np.asarray(lam_W, dtype=np.float64)
    regime = anchor['regime']
    valid = anchor['valid']
    wp = np.full_like(lam_W, np.nan, dtype=np.float64)
    wpp = np.full_like(lam_W, np.nan, dtype=np.float64)

    # Scat: sn parametrization, ℘ = e₃ + (e₁−e₃)/sn²(u, k²)
    m = (regime == 0) & valid
    if np.any(m):
        e1 = anchor['e1'][m]; e2 = anchor['e2'][m]; e3 = anchor['e3'][m]
        e13 = e1 - e3
        k2 = np.clip((e2 - e3) / e13, 0.0, 1.0 - 1e-15)
        sqrt_e13 = np.sqrt(np.maximum(e13, 0.0))
        u = lam_W[m] * sqrt_e13
        sn, cn, dn, _ = ellipj(u, k2)
        sn2 = sn * sn
        wp[m] = e3 + e13 / sn2
        wpp[m] = -2.0 * e13 ** 1.5 * cn * dn / (sn ** 3)

    # Rect: cn parametrization, ℘ = e_r + H·(1+cn)/(1−cn)
    m = (regime == 1) & valid
    if np.any(m):
        e_r = anchor['e_r'][m]; H = anchor['H'][m]; kp2 = anchor['kp2'][m]
        sqrtH = np.sqrt(np.maximum(H, 0.0))
        arg = 2.0 * sqrtH * lam_W[m]
        sn, cn, dn, _ = ellipj(arg, kp2)
        one_minus_cn = 1.0 - cn
        wp[m] = e_r + H * (1.0 + cn) / one_minus_cn
        # ℘'(λ) from the cn form: derive via chain rule.
        # d/dλ [H(1+cn)/(1-cn)] = H·[-sn·dn·(1-cn) - (1+cn)·sn·dn] / (1-cn)²
        #                      · d(u)/dλ = 2√H
        # = -2H·sn·dn / (1-cn)² · 2√H  (wait: let me redo)
        # Let u = 2√H·λ; d/dλ = 2√H·d/du.
        # d/du [(1+cn)/(1-cn)] = [−sn·dn·(1−cn) − (1+cn)·sn·dn]/(1−cn)²
        #                     = −sn·dn·[(1−cn) + (1+cn)] / (1−cn)²
        #                     = −2·sn·dn / (1−cn)²
        # ∴ ℘'(λ) = H · (−2·sn·dn / (1−cn)²) · 2√H
        #         = −4·H^{3/2}·sn·dn / (1−cn)²
        wpp[m] = -4.0 * (H ** 1.5) * sn * dn / (one_minus_cn ** 2)

    return wp, wpp


# ===========================================================================
# Inverse ℘: solve ℘(λ_W) = W for λ_W ∈ (0, ω) — regime-aware.
# ===========================================================================

def wp_inv_batch(W, anchor):
    """λ_W = ℘⁻¹(W) per ray, regime-masked via Carlson R_F.

    NaN on invalid or out-of-range rays.
    """
    W = np.asarray(W, dtype=np.float64)
    regime = anchor['regime']
    valid = anchor['valid']
    lam_W = np.full_like(W, np.nan, dtype=np.float64)

    m = (regime == 0) & valid & np.isfinite(W)
    if np.any(m):
        e1 = anchor['e1'][m]; e2 = anchor['e2'][m]; e3 = anchor['e3'][m]
        lam_W[m] = np.real(_wp_inv_uniform(W[m], e1, e2, e3))

    m = (regime == 1) & valid & np.isfinite(W)
    if np.any(m):
        e_r = anchor['e_r'][m]
        e_c1 = anchor['e_c1'][m]; e_c2 = anchor['e_c2'][m]
        lam_W[m] = np.real(_wp_inv_uniform(W[m], e_r, e_c1, e_c2))

    return lam_W


# ===========================================================================
# Polar anchor — vectorised polar Jacobi parameters
# ===========================================================================

def polar_anchor_batch(a, xi, eta, mu_obs):
    """Per-ray polar Jacobi anchor: μ₊, k², u_obs, √udiff, ...

    Parameters
    ----------
    a       : scalar.
    xi, eta : (N,) arrays.
    mu_obs  : scalar or (N,) array.

    Returns
    -------
    dict with (N,) arrays:
        up, um, udiff, sqrt_udiff, mu_plus, mu_plus_sq, k2,
        psi_obs, u_obs, K_comp, valid (bool)
    """
    a = float(a)
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    mu_obs = np.asarray(mu_obs, dtype=np.float64)
    if mu_obs.ndim == 0:
        mu_obs = np.broadcast_to(mu_obs, xi.shape).astype(np.float64)

    a2 = a * a
    Bc = eta + xi * xi - a2
    disc = Bc * Bc + 4.0 * a2 * eta
    sd = np.sqrt(np.maximum(disc, 0.0))

    # Stable up — avoid cancellation for Bc ≥ 0
    up = np.where((Bc >= 0.0) & ((sd + Bc) > 1e-300),
                  2.0 * a2 * eta / (sd + Bc + 1e-300),
                  0.5 * (-Bc + sd))
    um = -0.5 * (Bc + sd)
    udiff = up - um

    valid = (udiff > 0.0) & np.isfinite(udiff)
    with np.errstate(invalid='ignore', divide='ignore'):
        mu_plus_sq = np.where(a2 > 0.0, up / a2, 0.0)
        mu_plus = np.sqrt(np.maximum(mu_plus_sq, 0.0))
        k2 = np.clip(up / np.where(udiff > 0.0, udiff, 1.0), 0.0, 1.0 - 1e-15)
        sqrt_udiff = np.sqrt(np.maximum(udiff, 0.0))
        cos_psi_obs = np.clip(mu_obs / np.where(mu_plus > 0.0, mu_plus, 1.0),
                               -1.0, 1.0)
        psi_obs = np.arccos(cos_psi_obs)
        u_obs = ellipkinc(psi_obs, k2)
        K_comp = ellipk(k2)

    valid = valid & (mu_plus > 0.0) & np.isfinite(u_obs) & np.isfinite(K_comp)
    return dict(
        up=up, um=um, udiff=udiff, sqrt_udiff=sqrt_udiff,
        mu_plus=mu_plus, mu_plus_sq=mu_plus_sq,
        k2=k2, psi_obs=psi_obs, u_obs=u_obs, K_comp=K_comp,
        valid=valid,
    )
