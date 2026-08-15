"""Vectorised closed-form disk-return pipeline.

API:
  first_return(a, xi, eta, R0, s_r)  ->  (fate_code, lam1, R1)
     arrays of shape (N,).  fate_code ∈ {0:return, 1:escape, 2:plunge,
                                          3:marginal, 4:noncrossing}.

All steps are vectorised numpy ufuncs; the per-photon cost is amortised
across the batch.  The quartic and cubic roots are computed in closed
form (Ferrari/Cardano) so no iterative eigenvalue solver is called.

Handles both Weierstrass regimes uniformly:
  Δ_W > 0 (scattering, rhombic lattice): sn-based forward ℘.
  Δ_W < 0 (rectangular lattice): cn-based forward ℘.
For the inverse ℘⁻¹ we use Carlson's R_F (scipy.special.elliprf), which
works uniformly for both signs (complex conjugate args allowed).
"""

import numpy as np
from scipy.special import ellipk, ellipj, elliprf

FATE_RETURN, FATE_ESCAPE, FATE_PLUNGE, FATE_MARGINAL, FATE_NOCROSS = range(5)

# ---------------------------------------------------------------------
# Closed-form polynomial solvers
# ---------------------------------------------------------------------
def _cardano_three_real(p, q):
    """Three real roots of depressed cubic t^3 + p t + q = 0 (vectorised).
    Valid when p < 0 and -4p³ - 27q² ≥ 0.  Output sorted ascending.
    """
    r0 = np.sqrt(np.maximum(-p / 3.0, 0.0))
    # arg = (3q/2)/(p·r0) but guard against p·r0 → 0 (degenerate)
    pr0 = np.where(np.abs(p * r0) > 1e-300, p * r0, 1.0)
    arg = np.clip(1.5 * q / pr0, -1.0, 1.0)
    theta = np.arccos(arg) / 3.0
    t0 = 2.0 * r0 * np.cos(theta)
    t1 = 2.0 * r0 * np.cos(theta - 2.0 * np.pi / 3.0)
    t2 = 2.0 * r0 * np.cos(theta - 4.0 * np.pi / 3.0)
    stack = np.stack([t0, t1, t2], axis=0)
    stack.sort(axis=0)
    return stack[0], stack[1], stack[2]


def _cardano_one_real(p, q):
    """Single real root of depressed cubic t^3 + p t + q = 0 (vectorised).
    Valid when the other two roots are complex, i.e. q²/4 + p³/27 > 0.
    """
    disc = q * q / 4.0 + p ** 3 / 27.0
    sd = np.sqrt(np.maximum(disc, 0.0))
    u1 = -q / 2.0 + sd
    u2 = -q / 2.0 - sd
    def cbrt_signed(x):
        return np.sign(x) * np.abs(x) ** (1.0 / 3.0)
    return cbrt_signed(u1) + cbrt_signed(u2)


def _quartic_largest_real(p, q, r):
    """Largest REAL root of depressed quartic x^4 + p x² + q x + r = 0.
    Returns (R, pair_real_mask) where pair_real_mask is 2-tuple of bools
    (plus_is_real, minus_is_real) indicating which Ferrari pair is real.

    Vectorised.  Robust to Δ>0 (4 real) and Δ<0 (2 real, 2 complex).
    """
    # Resolvent cubic in m: m³ + p m² + (p²/4 − r) m − q²/8 = 0
    # Depress by m → t − p/3
    A = p
    B = p * p / 4.0 - r
    C = -q * q / 8.0
    P = B - A * A / 3.0
    Q = 2.0 * A ** 3 / 27.0 - A * B / 3.0 + C

    # Trigonometric (three real) branch
    three_real = (-4.0 * P ** 3 - 27.0 * Q ** 2) > 0
    r0 = np.sqrt(np.maximum(-P / 3.0, 0.0))
    pr0 = np.where(np.abs(P * r0) > 1e-300, P * r0, 1.0)
    arg = np.clip(1.5 * Q / pr0, -1.0, 1.0)
    theta = np.arccos(arg) / 3.0
    t1 = 2.0 * r0 * np.cos(theta)
    t2 = 2.0 * r0 * np.cos(theta - 2.0 * np.pi / 3.0)
    t3 = 2.0 * r0 * np.cos(theta - 4.0 * np.pi / 3.0)
    # Single-real branch (Cardano)
    u_arg = Q * Q / 4.0 + P ** 3 / 27.0
    u_sqrt = np.sqrt(np.maximum(u_arg, 0.0))
    def cbrt_signed(x):
        return np.sign(x) * np.abs(x) ** (1.0 / 3.0)
    t_single = cbrt_signed(-Q / 2.0 + u_sqrt) + cbrt_signed(-Q / 2.0 - u_sqrt)
    # Prefer the largest real root for numerical stability
    t_three = np.maximum(np.maximum(t1, t2), t3)
    t_chosen = np.where(three_real, t_three, t_single)
    m = t_chosen - A / 3.0
    # Safeguard m > 0 for alpha = √(2m); tiny values fall through as plunge
    m_pos = np.maximum(m, 1e-300)
    alpha = np.sqrt(2.0 * m_pos)
    beta_plus = p / 2.0 + m - q / (2.0 * alpha)
    beta_minus = p / 2.0 + m + q / (2.0 * alpha)

    D_plus = alpha * alpha - 4.0 * beta_plus
    D_minus = alpha * alpha - 4.0 * beta_minus

    plus_real = D_plus >= 0
    minus_real = D_minus >= 0
    sD_plus = np.sqrt(np.maximum(D_plus, 0.0))
    sD_minus = np.sqrt(np.maximum(D_minus, 0.0))

    # Largest root from each pair (if real)
    r_plus_max = 0.5 * (-alpha + sD_plus)
    r_minus_max = 0.5 * (alpha + sD_minus)
    # Use -inf for complex pairs so max skips them
    r_plus_max = np.where(plus_real, r_plus_max, -np.inf)
    r_minus_max = np.where(minus_real, r_minus_max, -np.inf)
    R_largest = np.maximum(r_plus_max, r_minus_max)
    return R_largest


def _quartic_sorted_real_roots(p, q, r):
    """All four real roots of x⁴ + p x² + q x + r = 0, sorted ascending.
    Vectorised.  Assumes the caller has verified the four-real branch
    (Δ_W>0 AND p<0 AND 4r − p² < 0).  Out-of-branch inputs get NaNs.

    Returns (r1, r2, r3, r4) with r1 ≤ r2 ≤ r3 ≤ r4.
    """
    A = p
    B = p * p / 4.0 - r
    C = -q * q / 8.0
    P = B - A * A / 3.0
    Q = 2.0 * A ** 3 / 27.0 - A * B / 3.0 + C

    # In the four-real case the resolvent cubic has three real roots;
    # pick the largest for numerical stability (gives m > 0).
    r0 = np.sqrt(np.maximum(-P / 3.0, 0.0))
    pr0 = np.where(np.abs(P * r0) > 1e-300, P * r0, 1.0)
    arg = np.clip(1.5 * Q / pr0, -1.0, 1.0)
    theta = np.arccos(arg) / 3.0
    t1 = 2.0 * r0 * np.cos(theta)
    t2 = 2.0 * r0 * np.cos(theta - 2.0 * np.pi / 3.0)
    t3 = 2.0 * r0 * np.cos(theta - 4.0 * np.pi / 3.0)
    t_best = np.maximum(np.maximum(t1, t2), t3)
    m = t_best - A / 3.0
    m_pos = np.maximum(m, 1e-300)
    alpha = np.sqrt(2.0 * m_pos)
    beta_plus = p / 2.0 + m - q / (2.0 * alpha)
    beta_minus = p / 2.0 + m + q / (2.0 * alpha)
    D_plus = alpha * alpha - 4.0 * beta_plus
    D_minus = alpha * alpha - 4.0 * beta_minus
    sD_plus = np.sqrt(np.maximum(D_plus, 0.0))
    sD_minus = np.sqrt(np.maximum(D_minus, 0.0))
    r_a = 0.5 * (-alpha + sD_plus)
    r_b = 0.5 * (-alpha - sD_plus)
    r_c = 0.5 * (alpha + sD_minus)
    r_d = 0.5 * (alpha - sD_minus)
    # If any D < 0 we are not actually four-real — fill with NaN
    bad = (D_plus < 0) | (D_minus < 0)
    stack = np.stack([r_a, r_b, r_c, r_d], axis=0)
    stack.sort(axis=0)
    r1, r2, r3, r4 = stack[0], stack[1], stack[2], stack[3]
    r1 = np.where(bad, np.nan, r1)
    r2 = np.where(bad, np.nan, r2)
    r3 = np.where(bad, np.nan, r3)
    r4 = np.where(bad, np.nan, r4)
    return r1, r2, r3, r4


# ---------------------------------------------------------------------
# Carlson R_F based inverse ℘ — uniform over Δ sign
# ---------------------------------------------------------------------
def _wp_inv_uniform(w, e1, e2, e3):
    """λ = ℘⁻¹(w), computed via Carlson's R_F.  Handles both Δ>0 (e1,e2,e3
    all real) and Δ<0 (e1 real, e2,e3 complex conjugates).
    Input arrays are broadcast together; output is real."""
    # scipy.special.elliprf broadcasts naturally; supports complex args.
    val = elliprf(w - e1, w - e2, w - e3)
    return val.real if np.iscomplexobj(val) else val


# ---------------------------------------------------------------------
# Forward ℘ for both regimes
# ---------------------------------------------------------------------
def _wp_forward_sn(lam, e1, e2, e3):
    """℘(λ) for Δ>0 via sn:  ℘ = e3 + (e1-e3)/sn²(λ√(e1-e3), m)."""
    m = (e2 - e3) / (e1 - e3)
    sqrt_e13 = np.sqrt(e1 - e3)
    sn, _, _, _ = ellipj(lam * sqrt_e13, m)
    return e3 + (e1 - e3) / (sn * sn)


def _wp_forward_cn(lam, e_real, H, kp2):
    """℘(λ) for Δ<0 via cn:  ℘ = e_r + H·(1+cn)/(1-cn), cn = cn(2√H λ, k_p)."""
    sqrtH = np.sqrt(H)
    _, cn, _, _ = ellipj(2.0 * sqrtH * lam, kp2)
    return e_real + H * (1.0 + cn) / (1.0 - cn)


# =====================================================================
# Main entry point
# =====================================================================
def first_return(a, xi, eta, R0, s_r):
    """
    Vectorised first-return computation for a batch of equatorial photons.

    Inputs: numpy arrays of identical shape.  s_r ∈ {+1, -1}.
    Output:
        fate : int8 array, values in {0..4}
        lam1 : float64 array, Mino time (since emission) of the first
               equator crossing (NaN if none)
        R1   : float64 array, BL radius at that crossing (NaN if none)
    """
    a = np.asarray(a, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    R0 = np.asarray(R0, dtype=np.float64)
    s_r = np.asarray(s_r, dtype=np.float64)

    shape = a.shape
    fate = np.full(shape, FATE_NOCROSS, dtype=np.int8)
    lam1 = np.full(shape, np.nan, dtype=np.float64)
    R1 = np.full(shape, np.nan, dtype=np.float64)

    # -----------------------------------------------------------------
    # Step 1: PD coordinates
    # -----------------------------------------------------------------
    xi_PD = a * (a - xi)
    eta_PD = eta + (xi - a) ** 2

    # -----------------------------------------------------------------
    # Step 2: Angular sector, Γ_p (same regardless of Δ sign)
    # -----------------------------------------------------------------
    Bc = eta_PD - 2.0 * xi_PD
    Cc = xi_PD * xi_PD - eta_PD * a * a
    disc_th = Bc * Bc - 4.0 * Cc
    disc_ok = disc_th >= 0
    sd = np.sqrt(np.where(disc_ok, disc_th, 0.0))
    up = 0.5 * (-Bc + sd)
    um = 0.5 * (-Bc - sd)
    angular_ok = disc_ok & (up > -1e-12) & (up > um + 1e-14)
    ddiff = np.where(angular_ok, up - um, 1.0)
    kp_sq_angular = np.where(angular_ok, up / ddiff, 0.0)
    Gp = 2.0 * ellipk(kp_sq_angular) / np.sqrt(np.maximum(ddiff, 1e-300))

    # -----------------------------------------------------------------
    # Step 3: Cieślik invariants
    # -----------------------------------------------------------------
    a2 = (2.0 * xi_PD - eta_PD) / 6.0
    a3 = eta_PD / 2.0  # this is c/4 where c = 2·η_PD
    a4 = xi_PD ** 2 - eta_PD * a * a
    g2 = a4 + 3.0 * a2 * a2
    g3 = a2 * a4 - a2 ** 3 - a3 * a3
    Delta_W = g2 ** 3 - 27.0 * g3 ** 2

    # Regime masks
    is_marg = Delta_W == 0
    is_scat = (Delta_W > 0) & angular_ok
    is_rect = (Delta_W < 0) & angular_ok
    fate = np.where(is_marg, FATE_MARGINAL, fate)

    # Quartic coefficients (depressed form: r⁴ + b r² + c r + d)
    b = 2.0 * xi_PD - eta_PD
    c = 2.0 * eta_PD
    d = xi_PD ** 2 - eta_PD * a * a

    # Horizon
    rp = 1.0 + np.sqrt(np.maximum(1.0 - a * a, 0.0))

    # -----------------------------------------------------------------
    # Zone sub-classification within Δ_W > 0:
    #   — four-real ψ_q:   p<0 AND 4s−p²<0
    #   — Descartes count of shifted quartic ψ_q(x+R₀) tells us where
    #     R₀ sits:  0 sign changes → OUTER, 2 → MIDDLE, 4 → INNER.
    # -----------------------------------------------------------------
    four_real = is_scat & (b < 0.0) & (4.0 * d - b * b < 0.0)
    psi_R0 = R0 ** 4 + b * R0 ** 2 + c * R0 + d
    c4_sh = np.ones_like(R0)
    c3_sh = 4.0 * R0
    c2_sh = 6.0 * R0 * R0 + b
    c1_sh = 4.0 * R0 ** 3 + 2.0 * b * R0 + c
    c0_sh = psi_R0
    sgn = np.stack([np.sign(c4_sh), np.sign(c3_sh), np.sign(c2_sh),
                    np.sign(c1_sh), np.sign(c0_sh)], axis=0)
    nsc = np.zeros(shape, dtype=np.int8)
    prev = sgn[0].copy()
    for k in range(1, 5):
        curr = sgn[k]
        bump = (curr != 0) & (prev != 0) & (curr != prev)
        nsc = nsc + bump.astype(np.int8)
        prev = np.where(curr != 0, curr, prev)

    is_middle = four_real & (nsc == 2) & (psi_R0 > 0)
    # Remove middle-zone photons from the outer scattering branch so the
    # outer-anchor Biermann formulas don't process them (they need a
    # different anchor at r₃, not r₄).
    is_scat_outer = is_scat & ~is_middle

    # ================================================================
    # Process each regime.  The two branches do the same pipeline
    # but use different cubic-root solvers and forward ℘ evaluators.
    # ================================================================

    def _core(mask, wp_forward_fn, e_triple_fn):
        """Process photons in `mask`.  e_triple_fn(g2m, g3m) returns
        three scalars per photon (e1, e2, e3 — real for scat, with
        e2, e3 complex conj for rect).  wp_forward_fn(lam, *e_triple)
        evaluates ℘(λ)."""
        if not mask.any():
            return
        g2m = g2[mask]; g3m = g3[mask]
        b_m = b[mask]; c_m = c[mask]; d_m = d[mask]
        R0_m = R0[mask]; rp_m = rp[mask]; s_r_m = s_r[mask]; Gp_m = Gp[mask]

        # Largest real root of ψ_q (outermost turning point)
        R_out = _quartic_largest_real(b_m, c_m, d_m)

        # Anchor validity: must have a real R_out ≥ 0 within numerical tol.
        # If R_out < 0 the emission is effectively in an unphysical region —
        # fall through to plunge (for rect) or noncrossing (for scat).
        valid = np.isfinite(R_out)

        # Cubic roots (in the sense needed by forward ℘)
        e1, e2, e3 = e_triple_fn(g2m, g3m)

        # Biermann anchor quantities
        phi_R = (12.0 * R_out * R_out + 2.0 * b_m) / 24.0
        psi_p = 4.0 * R_out ** 3 + 2.0 * b_m * R_out + c_m

        # Möbius images W(r)
        def W(r):
            return phi_R + psi_p / (4.0 * (r - R_out))

        W_R0 = W(R0_m)
        W_rp = W(rp_m)
        W_inf = phi_R  # W(∞) = φ(R)

        # Inverse ℘ via Carlson R_F
        mu_esc = _wp_inv_uniform(W_inf, e1, e2, e3)
        mu_e_abs = _wp_inv_uniform(W_R0, e1, e2, e3)
        mu_rp = _wp_inv_uniform(W_rp, e1, e2, e3)

        # Horizon-above-turning?  (plunge path for s_r=-1 if yes.)
        horizon_above = rp_m > R_out

        # Physical λ_emit in the outer-anchor parameterization:
        #  λ = 0  ↔ r = R_out (turning / pole of ℘)
        #  λ = +mu_esc ↔ r = ∞ (on the descending ℘ branch)
        # Outgoing (s_r=+1): λ_emit = +mu_e_abs ∈ (0, mu_esc), dr/dλ > 0.
        # Ingoing  (s_r=-1): λ_emit = −mu_e_abs ∈ (−mu_esc, 0), dr/dλ < 0.
        out_mask = s_r_m > 0
        in_mask = s_r_m < 0

        # Duration from emission to fate (escape or plunge):
        #   s_r=+1: to ∞              → mu_esc − mu_e_abs
        #   s_r=-1 & horizon_above:
        #           to horizon (before bounce) → mu_e_abs − mu_rp
        #   s_r=-1 & !horizon_above:
        #           bounce at R_out then ∞  → mu_esc + mu_e_abs
        dur = np.where(out_mask,
                       mu_esc - mu_e_abs,
                       np.where(horizon_above,
                                mu_e_abs - mu_rp,
                                mu_esc + mu_e_abs))

        # Final-fate if no return
        fate_final = np.where(out_mask, FATE_ESCAPE,
                              np.where(horizon_above, FATE_PLUNGE, FATE_ESCAPE))

        has_return = (Gp_m < dur) & valid

        # Lam-value (absolute, from turning-point reference) for the
        # FIRST equator crossing.  Equator crossings at λ = λ_emit + n·Γ_p
        # for n = 1, 2, … with λ_emit = +s_r·mu_e_abs.
        lam_cross_abs = s_r_m * mu_e_abs + Gp_m  # signed
        # ℘ is even in λ (real axis), so use |λ| for forward ℘ eval.
        wp_lam = wp_forward_fn(np.abs(lam_cross_abs), e1, e2, e3)
        R_n = R_out + psi_p / (4.0 * (wp_lam - phi_R))

        fate_sub = np.where(has_return, FATE_RETURN, fate_final).astype(np.int8)
        lam_sub = np.where(has_return, Gp_m, np.nan)
        R_sub = np.where(has_return, R_n, np.nan)

        fate_sub = np.where(valid, fate_sub, FATE_NOCROSS).astype(np.int8)

        fate[mask] = fate_sub
        lam1[mask] = lam_sub
        R1[mask] = R_sub

    # ----- Δ > 0 scattering -----
    def e_triple_scat(g2m, g3m):
        e3s, e2s, e1s = _cardano_three_real(-g2m / 4.0, -g3m / 4.0)
        return e1s, e2s, e3s

    def wp_fwd_scat(lam, e1, e2, e3):
        return _wp_forward_sn(lam, e1, e2, e3)

    # ----- Δ < 0 rectangular -----
    def e_triple_rect(g2m, g3m):
        e_r = _cardano_one_real(-g2m / 4.0, -g3m / 4.0)
        # Complex conjugate roots: (w² + e_r·w + (e_r² − g2/4)) = 0
        # → w = −e_r/2 ± i·√(3e_r² − g2)/2
        rad = np.sqrt(np.maximum(3.0 * e_r * e_r - g2m, 0.0)) / 2.0
        e_c1 = (-e_r / 2.0 + 1j * rad)
        e_c2 = e_c1.conjugate()
        return e_r, e_c1, e_c2

    def wp_fwd_rect(lam, e_r, e_c1, e_c2):
        # H² = (e_r − Re e_c)² + (Im e_c)² = 3·e_r² − g2/4   (always ≥ 0
        # when Δ_W < 0, so no masking is needed).
        H = np.sqrt((e_r - np.real(e_c1)) ** 2 + np.imag(e_c1) ** 2)
        # k_p² = 1/2 − 3·e_r/(4·H)
        kp2 = 0.5 - 3.0 * e_r / (4.0 * H)
        return _wp_forward_cn(lam, e_r, H, kp2)

    # =================================================================
    # MIDDLE ZONE (Δ_W > 0, four real roots, r₂ < R₀ < r₃)
    # Anchor at r₃ (upper turning point).  Radial motion is periodic
    # with period 2ω₁.  First equator crossing still at τ = Γ_p, but
    # we need the middle-zone anchor because the trajectory never
    # reaches r₄ or escapes to ∞.
    # =================================================================
    def _core_middle(mask):
        if not mask.any():
            return
        g2m = g2[mask]; g3m = g3[mask]
        b_m = b[mask]; c_m = c[mask]; d_m = d[mask]
        R0_m = R0[mask]; rp_m = rp[mask]; s_r_m = s_r[mask]; Gp_m = Gp[mask]

        # Four sorted real roots of ψ_q
        r1, r2, r3, r4 = _quartic_sorted_real_roots(b_m, c_m, d_m)
        valid = np.isfinite(r3) & np.isfinite(r2)

        # Anchor at r₃.  ψ'(r₃) < 0 in this branch (sign of ψ_q flips
        # from + to − at r₃ going upward through the four roots).
        R_anc = r3
        phi_R = (12.0 * R_anc * R_anc + 2.0 * b_m) / 24.0
        psi_p = 4.0 * R_anc ** 3 + 2.0 * b_m * R_anc + c_m

        # Weierstrass roots (rhombic branch: e1>e2>e3 all real)
        e3s, e2s, e1s = _cardano_three_real(-g2m / 4.0, -g3m / 4.0)
        e1, e2, e3 = e1s, e2s, e3s

        # Möbius image W(r) = φ(R) + ψ'(R)/[4(r−R)]
        def W(r):
            return phi_R + psi_p / (4.0 * (r - R_anc))

        W_R0 = W(R0_m)
        W_rp = W(rp_m)

        # Radial half-period  ω₁  =  R_F(0, e1−e2, e1−e3)
        half_omega = elliprf(0.0 * e1, e1 - e2, e1 - e3)
        if np.iscomplexobj(half_omega):
            half_omega = half_omega.real
        two_omega1 = 2.0 * half_omega

        # μ_e = ℘⁻¹(W(R₀))  — principal branch, 0 < μ_e < ω₁
        mu_e = _wp_inv_uniform(W_R0, e1, e2, e3)

        # Plunge detection.  r₂ ≥ rp → BOUND (τ_plunge = ∞).
        # r₂ < rp and rp < r₃ → PLUNGE on the ingoing leg.
        plunge_case = r2 < rp_m

        # For MIDDLE_PLUNGE, rp ∈ (r₂, r₃) so W(rp) ∈ (e₁, ∞) — real
        # elliprf on principal branch.  Guard BOUND case with W_R0
        # (safe, always > e₁) so elliprf gets real args.
        W_rp_safe = np.where(plunge_case, W_rp, W_R0)
        mu_rp = _wp_inv_uniform(W_rp_safe, e1, e2, e3)

        # First plunge time in physical Mino time τ:
        #   s_r=-1 (ingoing):  τ_p = μ_rp − μ_e
        #   s_r=+1 (outgoing): τ_p = μ_rp + μ_e  (bounce off r₃ then descend)
        tau_p = np.where(s_r_m < 0, mu_rp - mu_e, mu_rp + mu_e)
        tau_plunge = np.where(plunge_case, tau_p, np.inf)

        # First equator crossing at physical τ = Γ_p
        has_return = (Gp_m < tau_plunge) & valid

        # Anchor-frame λ at emission
        #   s_r=-1:  λ_anc(0) = μ_e      (ingoing half 0..ω₁)
        #   s_r=+1:  λ_anc(0) = 2ω₁−μ_e  (outgoing half ω₁..2ω₁)
        lam_emit = np.where(s_r_m < 0, mu_e, two_omega1 - mu_e)
        lam_1 = lam_emit + Gp_m

        # Forward ℘ (sn-based, period 2ω₁ — handles wrap naturally)
        wp_1 = _wp_forward_sn(lam_1, e1, e2, e3)
        R_1 = R_anc + psi_p / (4.0 * (wp_1 - phi_R))

        fate_final = np.where(plunge_case, FATE_PLUNGE, FATE_NOCROSS)
        fate_sub = np.where(has_return, FATE_RETURN, fate_final).astype(np.int8)
        fate_sub = np.where(valid, fate_sub, FATE_NOCROSS).astype(np.int8)
        lam_sub = np.where(has_return, Gp_m, np.nan)
        R_sub = np.where(has_return, R_1, np.nan)

        fate[mask] = fate_sub
        lam1[mask] = lam_sub
        R1[mask] = R_sub

    _core(is_scat_outer, wp_fwd_scat, e_triple_scat)
    _core(is_rect, wp_fwd_rect, e_triple_rect)
    _core_middle(is_middle)
    return fate, lam1, R1


# =====================================================================
# Multi-bounce extension of first_return
# =====================================================================
def multi_return(a, xi, eta, R0, s_r, N_max=8):
    """
    Sequence of equator crossings for a batch of equatorial photons.

    Generalises ``first_return`` to track the first N_max successive
    crossings of the equatorial plane along the same null geodesic.
    Useful for mirror-reflection / multi-bounce calculations in which
    the photon's conserved quantities (E, L_z, Q) are preserved across
    each crossing.

    The radial coordinate at the n-th crossing is

        R_n = R_anc + ψ'(R)/[4(℘(λ_n) − φ(R))],

    with the same anchors used by ``first_return`` and

        λ_n^anc = s_r · μ_e + n · Γ_p          (outer scattering / rect)
        λ_n^anc = λ_emit + n · Γ_p             (middle)

    where Γ_p is the polar half-period (Mino time between consecutive
    equator crossings).  The fate at each n is gated by

        n · Γ_p < dur_to_plunge_or_escape.

    Parameters
    ----------
    a, xi, eta, R0, s_r : array-like
        Same broadcast rules as ``first_return``.  s_r ∈ {+1, -1}.
    N_max : int, default 8
        Maximum number of crossings to track.

    Returns
    -------
    fates : int8 array, shape inputs.shape + (N_max,)
        ``fates[..., n-1]`` is FATE_RETURN if the n-th crossing
        occurred, otherwise the photon's final fate (FATE_ESCAPE /
        FATE_PLUNGE / FATE_NOCROSS / FATE_MARGINAL) carried forward.
        Once a photon has ended, the fate stays the same for all
        subsequent n.
    lams : float64 array, shape inputs.shape + (N_max,)
        ``lams[..., n-1]`` = Mino time (since emission) of the n-th
        crossing.  NaN once the photon has ended.
    Rs : float64 array, shape inputs.shape + (N_max,)
        ``Rs[..., n-1]`` = BL radius of the n-th crossing.  NaN once
        the photon has ended.

    Notes
    -----
    At n=1, ``(fates[..., 0], lams[..., 0], Rs[..., 0])`` matches
    ``first_return``'s output.  Subsequent crossings reuse the same
    pre-computed anchors per photon, so the marginal cost of each
    additional bounce is dominated by the ℘ forward evaluator.
    """
    N_max = int(N_max)
    a = np.asarray(a, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    R0 = np.asarray(R0, dtype=np.float64)
    s_r = np.asarray(s_r, dtype=np.float64)

    shape_in = a.shape
    shape_out = shape_in + (N_max,)

    fates = np.full(shape_out, FATE_NOCROSS, dtype=np.int8)
    lams = np.full(shape_out, np.nan, dtype=np.float64)
    Rs = np.full(shape_out, np.nan, dtype=np.float64)

    # ---- Same per-photon setup as first_return ----
    xi_PD = a * (a - xi)
    eta_PD = eta + (xi - a) ** 2
    Bc = eta_PD - 2.0 * xi_PD
    Cc = xi_PD * xi_PD - eta_PD * a * a
    disc_th = Bc * Bc - 4.0 * Cc
    disc_ok = disc_th >= 0
    sd = np.sqrt(np.where(disc_ok, disc_th, 0.0))
    up = 0.5 * (-Bc + sd)
    um = 0.5 * (-Bc - sd)
    angular_ok = disc_ok & (up > -1e-12) & (up > um + 1e-14)
    ddiff = np.where(angular_ok, up - um, 1.0)
    kp_sq_angular = np.where(angular_ok, up / ddiff, 0.0)
    Gp = 2.0 * ellipk(kp_sq_angular) / np.sqrt(np.maximum(ddiff, 1e-300))

    a2 = (2.0 * xi_PD - eta_PD) / 6.0
    a4 = xi_PD ** 2 - eta_PD * a * a
    g2 = a4 + 3.0 * a2 * a2
    g3 = a2 * a4 - a2 ** 3 - (eta_PD / 2.0) ** 2
    Delta_W = g2 ** 3 - 27.0 * g3 ** 2

    is_marg = Delta_W == 0
    is_scat = (Delta_W > 0) & angular_ok
    is_rect = (Delta_W < 0) & angular_ok
    if is_marg.any():
        fates[is_marg, :] = FATE_MARGINAL

    b = 2.0 * xi_PD - eta_PD
    c = 2.0 * eta_PD
    d = xi_PD ** 2 - eta_PD * a * a
    rp = 1.0 + np.sqrt(np.maximum(1.0 - a * a, 0.0))

    four_real = is_scat & (b < 0.0) & (4.0 * d - b * b < 0.0)
    psi_R0 = R0 ** 4 + b * R0 ** 2 + c * R0 + d
    c4_sh = np.ones_like(R0)
    c3_sh = 4.0 * R0
    c2_sh = 6.0 * R0 * R0 + b
    c1_sh = 4.0 * R0 ** 3 + 2.0 * b * R0 + c
    c0_sh = psi_R0
    sgn = np.stack([np.sign(c4_sh), np.sign(c3_sh), np.sign(c2_sh),
                    np.sign(c1_sh), np.sign(c0_sh)], axis=0)
    nsc = np.zeros(shape_in, dtype=np.int8)
    prev = sgn[0].copy()
    for k in range(1, 5):
        curr = sgn[k]
        bump = (curr != 0) & (prev != 0) & (curr != prev)
        nsc = nsc + bump.astype(np.int8)
        prev = np.where(curr != 0, curr, prev)
    is_middle = four_real & (nsc == 2) & (psi_R0 > 0)
    is_scat_outer = is_scat & ~is_middle

    # ----- Δ > 0 scattering / Δ < 0 rectangular --------
    def e_triple_scat(g2m, g3m):
        e3s, e2s, e1s = _cardano_three_real(-g2m / 4.0, -g3m / 4.0)
        return e1s, e2s, e3s

    def wp_fwd_scat(lam, e1, e2, e3):
        return _wp_forward_sn(lam, e1, e2, e3)

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

    # ------- Outer / rect core with per-photon anchor reuse -------
    def _core_multi(mask, wp_forward_fn, e_triple_fn):
        if not mask.any():
            return
        g2m = g2[mask]; g3m = g3[mask]
        b_m = b[mask]; c_m = c[mask]; d_m = d[mask]
        R0_m = R0[mask]; rp_m = rp[mask]; s_r_m = s_r[mask]; Gp_m = Gp[mask]

        R_out = _quartic_largest_real(b_m, c_m, d_m)
        valid = np.isfinite(R_out)

        e1, e2, e3 = e_triple_fn(g2m, g3m)
        phi_R = (12.0 * R_out * R_out + 2.0 * b_m) / 24.0
        psi_p = 4.0 * R_out ** 3 + 2.0 * b_m * R_out + c_m

        W_R0 = phi_R + psi_p / (4.0 * (R0_m - R_out))
        W_rp = phi_R + psi_p / (4.0 * (rp_m - R_out))
        W_inf = phi_R

        mu_esc = _wp_inv_uniform(W_inf, e1, e2, e3)
        mu_e_abs = _wp_inv_uniform(W_R0, e1, e2, e3)
        mu_rp = _wp_inv_uniform(W_rp, e1, e2, e3)

        horizon_above = rp_m > R_out
        out_mask = s_r_m > 0

        dur = np.where(out_mask,
                       mu_esc - mu_e_abs,
                       np.where(horizon_above,
                                mu_e_abs - mu_rp,
                                mu_esc + mu_e_abs))
        fate_final = np.where(out_mask, FATE_ESCAPE,
                              np.where(horizon_above, FATE_PLUNGE,
                                       FATE_ESCAPE))

        # Loop over crossing index.  All scalars/arrays above are
        # per-photon constants — only λ and ℘ are re-evaluated.
        for n in range(1, N_max + 1):
            lam_cross_abs = s_r_m * mu_e_abs + n * Gp_m
            still_alive = (n * Gp_m < dur) & valid

            wp_lam = wp_forward_fn(np.abs(lam_cross_abs), e1, e2, e3)
            R_n = R_out + psi_p / (4.0 * (wp_lam - phi_R))

            fate_n = np.where(still_alive, FATE_RETURN, fate_final)
            fate_n = np.where(valid, fate_n, FATE_NOCROSS).astype(np.int8)

            lam_n = np.where(still_alive, n * Gp_m, np.nan)
            R_n_safe = np.where(still_alive, R_n, np.nan)

            fates[..., n - 1][mask] = fate_n
            lams[..., n - 1][mask]  = lam_n
            Rs[..., n - 1][mask]    = R_n_safe

    # ------- Middle-zone core with per-photon anchor reuse -------
    def _core_middle_multi(mask):
        if not mask.any():
            return
        g2m = g2[mask]; g3m = g3[mask]
        b_m = b[mask]; c_m = c[mask]; d_m = d[mask]
        R0_m = R0[mask]; rp_m = rp[mask]; s_r_m = s_r[mask]; Gp_m = Gp[mask]

        r1, r2, r3, r4 = _quartic_sorted_real_roots(b_m, c_m, d_m)
        valid = np.isfinite(r3) & np.isfinite(r2)

        R_anc = r3
        phi_R = (12.0 * R_anc * R_anc + 2.0 * b_m) / 24.0
        psi_p = 4.0 * R_anc ** 3 + 2.0 * b_m * R_anc + c_m

        e3s, e2s, e1s = _cardano_three_real(-g2m / 4.0, -g3m / 4.0)
        e1, e2, e3 = e1s, e2s, e3s

        W_R0 = phi_R + psi_p / (4.0 * (R0_m - R_anc))
        W_rp = phi_R + psi_p / (4.0 * (rp_m - R_anc))

        half_omega = elliprf(0.0 * e1, e1 - e2, e1 - e3)
        if np.iscomplexobj(half_omega):
            half_omega = half_omega.real
        two_omega1 = 2.0 * half_omega

        mu_e = _wp_inv_uniform(W_R0, e1, e2, e3)
        plunge_case = r2 < rp_m
        W_rp_safe = np.where(plunge_case, W_rp, W_R0)
        mu_rp = _wp_inv_uniform(W_rp_safe, e1, e2, e3)

        tau_p = np.where(s_r_m < 0, mu_rp - mu_e, mu_rp + mu_e)
        tau_plunge = np.where(plunge_case, tau_p, np.inf)

        lam_emit = np.where(s_r_m < 0, mu_e, two_omega1 - mu_e)
        fate_final = np.where(plunge_case, FATE_PLUNGE, FATE_NOCROSS)

        for n in range(1, N_max + 1):
            still_alive = (n * Gp_m < tau_plunge) & valid
            lam_n_anc = lam_emit + n * Gp_m

            wp_n = _wp_forward_sn(lam_n_anc, e1, e2, e3)
            R_n = R_anc + psi_p / (4.0 * (wp_n - phi_R))

            fate_n = np.where(still_alive, FATE_RETURN, fate_final)
            fate_n = np.where(valid, fate_n, FATE_NOCROSS).astype(np.int8)

            lam_n = np.where(still_alive, n * Gp_m, np.nan)
            R_n_safe = np.where(still_alive, R_n, np.nan)

            fates[..., n - 1][mask] = fate_n
            lams[..., n - 1][mask]  = lam_n
            Rs[..., n - 1][mask]    = R_n_safe

    _core_multi(is_scat_outer, wp_fwd_scat, e_triple_scat)
    _core_multi(is_rect, wp_fwd_rect, e_triple_rect)
    _core_middle_multi(is_middle)

    return fates, lams, Rs


# =====================================================================
# BRUTE-FORCE VERIFICATION against Dormand-Prince integration
# =====================================================================
def brute_first_return(a, xi, eta, R0, s_r,
                       rtol=1e-12, atol=1e-13, lam_max=50.0):
    """Integrate Kerr null geodesic from (R0, π/2) with p_θ = +√η and
    r' = s_r·√R_val.  Return (fate, lam1, R1)."""
    from scipy.integrate import solve_ivp
    def rhs(lam, y):
        r, th, rd, td = y
        Delta = r * r - 2 * r + a * a
        P = r * r + a * a - a * xi
        Pp = 2 * r
        Delta_p = 2 * r - 2
        Rp = 2 * P * Pp - Delta_p * ((xi - a) ** 2 + eta)
        s = np.sin(th); cc = np.cos(th)
        Tp = (-2 * cc * s * (a * a - xi * xi / (s * s))
              + 2 * xi * xi * cc ** 3 / s ** 3)
        return [rd, td, 0.5 * Rp, 0.5 * Tp]

    P0 = R0 * R0 + a * a - a * xi
    Delta0 = R0 * R0 - 2 * R0 + a * a
    R_val = P0 ** 2 - Delta0 * ((xi - a) ** 2 + eta)
    Theta_val = eta
    if R_val < 0:
        return FATE_NOCROSS, np.nan, np.nan
    rh = 1.0 + np.sqrt(1 - a * a)

    def hit_horizon(lam, y):  return y[0] - (rh + 1e-5)
    hit_horizon.terminal = True; hit_horizon.direction = -1

    def equator(lam, y):      return y[1] - np.pi / 2 - 1e-14
    equator.terminal = False

    sol = solve_ivp(rhs, [0, lam_max],
                    [R0, np.pi / 2, s_r * np.sqrt(R_val), +np.sqrt(Theta_val)],
                    method="DOP853", rtol=rtol, atol=atol,
                    events=[hit_horizon, equator], max_step=0.01,
                    dense_output=True)
    crossings = sol.t_events[1]
    if len(crossings) >= 1:
        lam1 = crossings[crossings > 1e-10]
        if len(lam1) >= 1:
            lam1 = float(lam1[0])
            r1 = float(sol.sol(lam1)[0])
            return FATE_RETURN, lam1, r1
    if sol.status == 1:   # horizon hit
        return FATE_PLUNGE, np.nan, np.nan
    return FATE_ESCAPE, np.nan, np.nan


# =====================================================================
# SELF-TEST
# =====================================================================
