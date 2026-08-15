"""PhotonGeodesicBatch — vectorised counterpart to PhotonGeodesic.

Holds per-ray conserved quantities and initial-event data as NumPy
arrays and evaluates the forward closed-form trajectory on the whole
batch in a single broadcast op.

Shared scalars: the Kerr spin `a`, and — for observers that fire all
rays from a single event — `r0, θ0, t0, φ0`.  These can also be
per-ray arrays if a future distributed-observer class requires it.

Per-ray arrays: `xi, eta, s_r0, s_theta0`, each of shape (N,).

Public API mirrors the scalar PhotonGeodesic, but every method takes
and returns arrays:

    r_of_lam(lam)             -> (N,) array
    theta_of_lam(lam)         -> (N,) array
    lam_of_r(r_target, n=0)   -> (N,) array, NaN on misses
    lam_of_theta(θ_target, n=0) -> (N,) array, NaN on misses
    hit_event_rtheta(lam)     -> dict of (N,) arrays
    hit_event(lam)            -> dict of (N,) arrays  [Phase 4c]

This module implements the Phase 4b primitives (geometry-only).  The
(t, φ) closed-form (Phase 4c) will add vectorised G_τ, G_σ, J_τ, J_σ.
"""

import numpy as np
from scipy.special import ellipj, ellipkinc

from ._batch_primitives import (
    radial_anchor_batch, wp_forward_batch, wp_inv_batch, polar_anchor_batch,
)
from .momentum import photon_p_contravariant, photon_p_covariant


class PhotonGeodesicBatch:
    """Batch of N Kerr photon geodesics sharing (a, r₀, θ₀).

    Parameters
    ----------
    a        : scalar — Kerr spin.
    xi, eta  : (N,) arrays — conserved quantities per ray.
    r0       : scalar — observer radius (shared).
    theta0   : scalar — observer polar angle (shared).
    t0, phi0 : scalars — observer event coords (shared; default 0, 0).
    s_r0     : (N,) array of ±1 — radial sign at the start point in the
               geodesic-integration (Mino-time) sense, i.e. dr/dλ at λ=0.
    s_theta0 : (N,) array of ±1 — polar sign at the start point in the
               geodesic-integration (Mino-time) sense, i.e. dθ/dλ at λ=0.
    trace_direction : ±1, default +1
               Relationship between the geodesic-integration Mino time
               and the photon's PHYSICAL forward time.

                 +1 ("forward emission") — λ increases in the same
                     direction as the photon's physical time.  The
                     `s_r0` / `s_theta0` arguments are the FORWARD-
                     PHYSICAL signs at λ=0.  Used by `DiskEmitter`.

                 −1 ("backward observer trace") — λ increases in the
                     OPPOSITE direction to the photon's physical time;
                     we trace the back-pointing geodesic from the
                     observer toward the source.  The `s_r0` /
                     `s_theta0` arguments are still Mino-time signs
                     (so the radial / polar inversion machinery works
                     unchanged), but the FORWARD-PHYSICAL k^μ at any
                     point is the negative of the spatial part of
                     `photon_p_contravariant(...)`.  Used by
                     `AsymptoticObserver` and `StationaryObserver`.

               `p_at_observer` and `p_at_hit` apply this conversion
               internally so that **every public 4-momentum returned
               by the batch is forward-physical** regardless of which
               observer fed it (POLARIZATION_CONVENTIONS.md §4).

    Notes
    -----
    The geodesic-integration internals (`r_of_lam`, `lam_of_r`,
    `radial_anchor`, polar machinery, hit_event, …) all operate in the
    Mino convention and are unchanged.  The `trace_direction` flag is
    consumed only at the photon-momentum output boundary.
    """

    def __init__(self, a, xi, eta, r0, theta0, t0, phi0, s_r0, s_theta0,
                 trace_direction=+1):
        self.a = float(a)
        self.xi = np.asarray(xi, dtype=np.float64)
        self.eta = np.asarray(eta, dtype=np.float64)
        self.r0 = float(r0)
        self.theta0 = float(theta0)
        self.t0 = float(t0)
        self.phi0 = float(phi0)
        self.s_r0 = np.asarray(s_r0, dtype=np.float64).astype(np.int8)
        self.s_theta0 = np.asarray(s_theta0, dtype=np.float64).astype(np.int8)
        td = int(trace_direction)
        if td not in (+1, -1):
            raise ValueError(
                f"trace_direction must be +1 or -1, got {trace_direction!r}"
            )
        self.trace_direction = td
        self.N = int(self.xi.size)
        self._rad = None
        self._pol = None

    # -------------------- intrinsics (lazy) ------------------------------

    def radial_anchor(self):
        if self._rad is None:
            self._rad = radial_anchor_batch(self.a, self.xi, self.eta, self.r0)
        return self._rad

    def polar_anchor(self):
        if self._pol is None:
            mu_obs = np.cos(self.theta0)
            self._pol = polar_anchor_batch(self.a, self.xi, self.eta, mu_obs)
        return self._pol

    # -------------------- radial: forward r(λ) and λ⁻¹(r) ----------------

    def _lam_W_of_tau(self, tau):
        """Map forward Mino time τ → Biermann parameter λ_W (per ray).

        s_r0 = −1 (inward):  λ_W(τ) = |λ_obs − τ|
        s_r0 = +1 (outward): λ_W(τ) = λ_obs + τ
        """
        anc = self.radial_anchor()
        lam_obs_W = anc['lam_obs_W']
        tau = np.asarray(tau, dtype=np.float64)
        tau_b = np.broadcast_to(tau, lam_obs_W.shape)
        return np.where(self.s_r0 < 0,
                        np.abs(lam_obs_W - tau_b),
                        lam_obs_W + tau_b)

    def r_of_lam(self, tau):
        """Forward Boyer-Lindquist r at Mino time τ ≥ 0 (per ray)."""
        anc = self.radial_anchor()
        lam_W = self._lam_W_of_tau(tau)
        wp, _ = wp_forward_batch(lam_W, anc)
        return anc['R_out'] + anc['psi_p'] / (4.0 * (wp - anc['phi_R']))

    def _lam_W_of_r(self, r_target):
        """λ_W corresponding to BL radius r_target (per ray)."""
        anc = self.radial_anchor()
        r_target = np.asarray(r_target, dtype=np.float64)
        r_t = np.broadcast_to(r_target, anc['R_out'].shape)
        with np.errstate(invalid='ignore', divide='ignore'):
            W = anc['phi_R'] + anc['psi_p'] / (4.0 * (r_t - anc['R_out']))
        return wp_inv_batch(W, anc)

    def lam_of_r(self, r_target, n_crossing=0):
        """Forward Mino time of the n-th r-crossing at r_target (per ray).

        Mirrors the scalar `lam_of_r` semantics:
            s_r0 = -1:
               n=0 = pre-bounce   (only if r_target > R_out)
               n=1 = post-bounce
            s_r0 = +1:
               n=0 = single outward crossing (only if r_target > r_obs)
        """
        anc = self.radial_anchor()
        R_out = anc['R_out']
        lam_obs_W = anc['lam_obs_W']
        r_target = float(r_target)
        N = self.N
        result = np.full(N, np.nan, dtype=np.float64)

        valid = anc['valid'] & (r_target >= R_out - 1e-10)
        # Near-turning: r_target ≈ R_out ⇒ λ_W_target = 0
        near_turning = valid & (np.abs(r_target - R_out) < 1e-12)
        elsewhere = valid & ~near_turning

        lam_W_t = np.full(N, np.nan, dtype=np.float64)
        if np.any(near_turning):
            lam_W_t[near_turning] = 0.0
        if np.any(elsewhere):
            r_arr = np.full(N, r_target, dtype=np.float64)
            lam_W_t[elsewhere] = self._lam_W_of_r_at(r_arr, elsewhere)

        # Ingoing rays (s_r0 = −1)
        in_mask = (self.s_r0 < 0) & valid
        if np.any(in_mask):
            pre = lam_W_t < lam_obs_W - 1e-14
            if n_crossing == 0:
                m = in_mask & pre
                result[m] = lam_obs_W[m] - lam_W_t[m]
                m = in_mask & ~pre
                result[m] = lam_obs_W[m] + lam_W_t[m]
            elif n_crossing == 1:
                # only exists for pre-bounce cases
                m = in_mask & pre
                result[m] = lam_obs_W[m] + lam_W_t[m]
            # n>1: no crossings in this regime

        # Outgoing rays (s_r0 = +1)
        out_mask = (self.s_r0 > 0) & valid
        if np.any(out_mask):
            reachable = out_mask & (lam_W_t > lam_obs_W + 1e-14)
            if n_crossing == 0:
                result[reachable] = lam_W_t[reachable] - lam_obs_W[reachable]

        return result

    def _lam_W_of_r_at(self, r_arr, mask):
        """Compute λ_W for the masked subset where r may be different per ray."""
        anc = self.radial_anchor()
        with np.errstate(invalid='ignore', divide='ignore'):
            W = anc['phi_R'] + anc['psi_p'] / (4.0 * (r_arr - anc['R_out']))
        # Build a per-ray anchor view by passing the masked subset.
        sub_anchor = dict(
            regime=anc['regime'][mask],
            valid=anc['valid'][mask],
            e1=anc['e1'][mask], e2=anc['e2'][mask], e3=anc['e3'][mask],
            e_r=anc['e_r'][mask],
            e_c1=anc['e_c1'][mask], e_c2=anc['e_c2'][mask],
        )
        return wp_inv_batch(W[mask], sub_anchor)

    # -------------------- polar: forward θ(λ) and λ⁻¹(θ) ----------------

    def _u_of_tau(self, tau):
        """Unfolded Jacobi u(τ) = u_obs + s_θ0 · τ · √udiff, per ray."""
        pol = self.polar_anchor()
        tau = np.asarray(tau, dtype=np.float64)
        tau_b = np.broadcast_to(tau, pol['u_obs'].shape)
        return pol['u_obs'] + self.s_theta0 * tau_b * pol['sqrt_udiff']

    def theta_of_lam(self, tau):
        """Forward polar θ at Mino time τ ≥ 0 (per ray)."""
        pol = self.polar_anchor()
        u = self._u_of_tau(tau)
        sn, cn, dn, _ = ellipj(u, pol['k2'])
        mu = pol['mu_plus'] * cn
        return np.arccos(np.clip(mu, -1.0, 1.0))

    def lam_of_theta(self, theta_target, n_crossing=0):
        """Mino time of the n-th polar crossing at θ_target (per ray)."""
        pol = self.polar_anchor()
        N = self.N
        theta_target = float(theta_target)
        mu_target = np.cos(theta_target)
        mu_plus = pol['mu_plus']
        # Admissibility: |μ_target| ≤ μ_+
        admissible = pol['valid'] & (np.abs(mu_target) <= mu_plus + 1e-14)
        result = np.full(N, np.nan, dtype=np.float64)

        # F_target = F(ψ_target, k²)  where cos(ψ_target) = μ_target/μ_+
        cos_psi_t = np.clip(mu_target / np.where(mu_plus > 0.0, mu_plus, 1.0),
                             -1.0, 1.0)
        psi_target = np.arccos(cos_psi_t)
        F_target = ellipkinc(psi_target, pol['k2'])
        K = pol['K_comp']
        u_obs = pol['u_obs']
        s = self.s_theta0.astype(np.float64)
        sqrt_udiff = pol['sqrt_udiff']

        # cn(u, k²) = μ_target/μ_+  ⇔  u ∈ { ±F_target + 4·K·n | n ∈ ℤ }.
        # For τ ≥ 0 we need sign(u − u_obs) = s.
        # Build a (N, 26) grid of candidates and pick the n-th positive per ray.
        n_range = np.arange(-6, 7)
        # Shape (N, 2, 13):  sign × n
        sign_axis = np.array([+1.0, -1.0])[None, :, None]          # (1,2,1)
        n_axis = n_range[None, None, :]                             # (1,1,13)
        F_col = F_target[:, None, None]                             # (N,1,1)
        K_col = K[:, None, None]                                    # (N,1,1)
        u_c = sign_axis * F_col + 4.0 * K_col * n_axis              # (N,2,13)
        u_obs_col = u_obs[:, None, None]
        s_col = s[:, None, None]
        sqrt_udiff_col = sqrt_udiff[:, None, None]
        # dτ = (u_c - u_obs) / (s · √udiff)
        with np.errstate(invalid='ignore', divide='ignore'):
            dtau = (u_c - u_obs_col) / (s_col * sqrt_udiff_col)
        dtau = dtau.reshape(N, -1)                                  # (N,26)
        # Mask non-positive or non-finite entries out with +inf so argsort puts them last
        valid_dtau = np.where((dtau > 1e-14) & np.isfinite(dtau), dtau, np.inf)
        sorted_dtau = np.sort(valid_dtau, axis=1)
        pick = sorted_dtau[:, n_crossing] if n_crossing < sorted_dtau.shape[1] else np.full(N, np.inf)
        result = np.where(admissible & np.isfinite(pick), pick, np.nan)
        return result

    # -------------------- geometry-only event ----------------------------

    def hit_event_rtheta(self, lam, r_at_lam=None):
        """Full per-ray geometry-only hit event.

        Parameters
        ----------
        lam     : (N,) array — forward Mino time per ray (NaN for misses).
        r_at_lam: (N,) array or None — r(λ) if pre-computed (e.g. the
                  shell target already knows r_target); else we call
                  r_of_lam(lam) internally.

        Returns
        -------
        dict of (N,) arrays with keys t_hit, phi_hit, r_hit, theta_hit,
        lam_hit, tau_hit, sigma_hit, G_tau, G_sig, J_tau, J_sig,
        s_r_at_hit.  t_hit, phi_hit, tau, sigma, G, J are all NaN.
        """
        lam = np.asarray(lam, dtype=np.float64)
        N = self.N
        miss = ~np.isfinite(lam) | (lam < 0.0)
        # Work only on valid rays
        r_hit = np.full(N, np.nan, dtype=np.float64)
        theta_hit = np.full(N, np.nan, dtype=np.float64)
        s_r_at_hit = np.zeros(N, dtype=np.int8)
        if (~miss).any():
            good = ~miss
            if r_at_lam is None:
                r_hit[good] = self.r_of_lam(lam)[good]
            else:
                r_at_lam_arr = np.asarray(r_at_lam, dtype=np.float64)
                if r_at_lam_arr.ndim == 0:
                    r_at_lam_arr = np.full(N, float(r_at_lam_arr))
                r_hit[good] = r_at_lam_arr[good]
            theta_hit[good] = self.theta_of_lam(lam)[good]
            # s_r_at_hit: pre- vs post-bounce for s_r0=−1 scat; monotone else.
            anc = self.radial_anchor()
            is_scat = (anc['regime'] == 0)
            pre = good & (self.s_r0 < 0) & is_scat & (lam < anc['lam_obs_W'] - 1e-14)
            post = good & (self.s_r0 < 0) & is_scat & ~pre
            rect_in = good & (self.s_r0 < 0) & ~is_scat
            out = good & (self.s_r0 > 0)
            s_r_at_hit[pre] = -1
            s_r_at_hit[post] = +1
            s_r_at_hit[rect_in] = -1
            s_r_at_hit[out] = +1

        lam_out = np.where(miss, np.nan, lam)
        return dict(
            t_hit=np.full(N, np.nan),
            phi_hit=np.full(N, np.nan),
            r_hit=r_hit,
            theta_hit=theta_hit,
            lam_hit=lam_out,
            tau_hit=np.full(N, np.nan),
            sigma_hit=np.full(N, np.nan),
            G_tau=np.full(N, np.nan), G_sig=np.full(N, np.nan),
            J_tau=np.full(N, np.nan), J_sig=np.full(N, np.nan),
            s_r_at_hit=s_r_at_hit,
        )

    # -------------------- full (t, r, θ, φ) event (Phase 4c) -------------

    def hit_event(self, lam, r_at_lam=None, s_r_at_lam=None, regularize=False):
        """Vectorised full Boyer-Lindquist event (t, r, θ, φ) at Mino λ.

        Parameters
        ----------
        lam        : (N,) array — forward Mino time per ray (NaN for misses).
        r_at_lam   : (N,) array or None — r(λ) if pre-computed; else
                     computed internally via r_of_lam.
        s_r_at_lam : (N,) array of ±1 or None — if None, inferred from
                     the regime and pre/post bounce diagnosis.
        regularize : bool — subtract asymptotic linear + log pieces of
                     t_hit (Gralla-Lupsasca convention).

        Returns
        -------
        dict of (N,) arrays keyed t_hit, phi_hit, r_hit, theta_hit,
        lam_hit, tau_hit, sigma_hit, G_tau, G_sig, J_tau, J_sig,
        s_r_at_hit.  Non-hit rays carry NaN.

        TODO (known bug — see KNOWN_ISSUES.md):
            phi_hit can come back as NaN on a measure-zero set of
            borderline cells even when lam, r_at_lam, s_r_at_lam are
            all finite and well-defined.  Symptom: the
            test_returning_radiation reflection-symmetry test
            reports |Δφ_land| = NaN under μ → −μ while |Δr_land| and
            |Δg_factor| are bit-clean.  Likely sources are the
            Π̂-period unfolded primitive when `lam` lands within
            ~1e-12 of an exact polar quarter-period K(k²), and
            Carlson R_J (scipy.special.elliprj) at radial turning
            points where one argument is at zero.  For Schwarzschild-
            proxy spins the PD-split σ = φ/a reconstruction also has
            large 1/a cancellations that exacerbate this.

            A `[KNOWN-FAIL]` marker on the reflection test guards
            the test suite; DiskEmitter._forward_fate's defensive
            intersection (first_return + lam_of_theta finite) keeps
            obviously bad cells out of the integrand.  When this is
            fixed, both can be removed.
        """
        from ._batch_tphi import polar_G_batch, radial_J_batch

        lam = np.asarray(lam, dtype=np.float64)
        N = self.N
        miss = ~np.isfinite(lam) | (lam < 0.0)
        good = ~miss

        r_hit = np.full(N, np.nan, dtype=np.float64)
        theta_hit = np.full(N, np.nan, dtype=np.float64)
        s_r_at_hit = np.zeros(N, dtype=np.int8)

        if good.any():
            if r_at_lam is None:
                r_hit[good] = self.r_of_lam(lam)[good]
            else:
                r_at_lam_arr = np.asarray(r_at_lam, dtype=np.float64)
                if r_at_lam_arr.ndim == 0:
                    r_at_lam_arr = np.full(N, float(r_at_lam_arr))
                r_hit[good] = r_at_lam_arr[good]
            theta_hit[good] = self.theta_of_lam(lam)[good]

        # s_r_at_hit: pre/post-bounce diagnosis (matches scalar hit_event)
        anc = self.radial_anchor()
        is_scat = (anc['regime'] == 0)
        if s_r_at_lam is None:
            pre_bounce = good & (self.s_r0 < 0) & is_scat \
                          & (lam < anc['lam_obs_W'] - 1e-14)
            post_bounce = good & (self.s_r0 < 0) & is_scat & ~pre_bounce
            rect_in = good & (self.s_r0 < 0) & ~is_scat
            out = good & (self.s_r0 > 0)
            s_r_at_hit[pre_bounce] = -1
            s_r_at_hit[post_bounce] = +1
            s_r_at_hit[rect_in] = -1
            s_r_at_hit[out] = +1
        else:
            s_r_at_lam_arr = np.asarray(s_r_at_lam, dtype=np.int8)
            s_r_at_hit[good] = np.sign(s_r_at_lam_arr[good]).astype(np.int8)
            s_r_at_hit[good & (s_r_at_hit == 0)] = +1

        # s_r_disk convention: +1 if s_r_at_hit has same sign as s_r0
        # (direct leg, no pericenter bounce between observer and hit),
        # -1 otherwise (bounced through pericenter en route to hit).
        s_r_disk = np.where(s_r_at_hit * self.s_r0 > 0, +1, -1).astype(np.float64)

        # ---- polar sector: observer → hit ------------------------------
        mu_obs = np.cos(self.theta0)
        s_mu_fwd = self.s_theta0.astype(np.float64)
        mu_obs_arr = np.full(N, mu_obs, dtype=np.float64)
        G_tau, G_sig = polar_G_batch(self.a, self.xi, self.eta,
                                      mu_obs_arr, s_mu_fwd, lam)

        # ---- radial sector: r_obs → r_hit ------------------------------
        # radial_J_batch internally runs only on rays where the closed form
        # is well-defined; we pass r_hit (NaN on miss rays) and let the
        # internal np.where mask non-finite results.
        r_obs_arr = np.full(N, self.r0, dtype=np.float64)
        J_tau, J_sig = radial_J_batch(anc, r_hit, r_obs_arr,
                                        s_r_disk, regularize=regularize)

        # Outgoing-ray sign flip: J_integrals_radial_closed returns
        #   I = ∫_{r_hit}^{r_obs} f(r) dr   on the direct leg (s_r_disk=+1).
        # That has the correct sign for an inward observer reaching
        # r_hit < r_obs, but is negated when r_hit > r_obs (outgoing ray
        # reaching an outer hit).  Bounce legs (s_r_disk=−1) are
        # integrated outward from pericenter in both sub-legs, so no
        # sign flip there.
        flip = (s_r_disk > 0) & (r_hit > self.r0)
        J_tau = np.where(flip, -J_tau, J_tau)
        J_sig = np.where(flip, -J_sig, J_sig)

        # Handle a = 0: σ = φ/a is undefined; at a = 0 the PD split is
        # singular.  The scalar path uses `_schwarzschild_tphi_quad`
        # instead; the batch path raises.
        if self.a == 0.0:
            raise NotImplementedError(
                "PhotonGeodesicBatch.hit_event does not yet support a = 0 "
                "(PD split τ = t − aφ, σ = φ/a is singular).")

        # ---- assemble ---------------------------------------------------
        tau = J_tau - G_tau
        sigma = J_sig + G_sig
        t_hit = self.t0 + tau + (self.a * self.a) * sigma
        phi_hit = self.phi0 + self.a * sigma

        lam_out = np.where(miss, np.nan, lam)
        # Mask all integral-derived quantities to NaN on non-hit rays.
        for arr in (t_hit, phi_hit, tau, sigma, G_tau, G_sig, J_tau, J_sig):
            arr[miss] = np.nan

        # Polar tangent sign at the hit (needed for p^θ).  Derived from
        # the unfolded Jacobi argument: s_θ(λ) = s_θ0 · sign(sn(u(λ), k²)).
        s_theta_at_hit = self._s_theta_at(lam, miss_mask=miss)

        return dict(
            t_hit=t_hit,
            phi_hit=phi_hit,
            r_hit=r_hit,
            theta_hit=theta_hit,
            lam_hit=lam_out,
            tau_hit=tau,
            sigma_hit=sigma,
            G_tau=G_tau, G_sig=G_sig,
            J_tau=J_tau, J_sig=J_sig,
            s_r_at_hit=s_r_at_hit,
            s_theta_at_hit=s_theta_at_hit,
        )

    # -------------------- polar tangent sign (for p^θ) -------------------

    def _s_theta_at(self, lam, miss_mask=None):
        """Sign of dθ/dλ at forward Mino time λ (per ray).

        From μ = μ_+ cn(u, k²) with u = u_obs + s_θ0 · λ · √udiff:
          dμ/du    = −μ_+ sn(u) dn(u)
          dμ/dλ    = (dμ/du)(du/dλ) = −μ_+ sn(u) dn(u) · s_θ0 √udiff
          dθ/dλ    = −dμ/dλ / sin θ   (sin θ > 0 on the open interval)
          s_θ(λ)   = sign(dθ/dλ) = s_θ0 · sign(sn(u))

        At turning points sn = 0 and p^θ = 0; we return s_θ0 by
        convention (the photon is about to re-reverse polar direction).
        """
        lam = np.asarray(lam, dtype=np.float64)
        N = self.N
        if miss_mask is None:
            miss_mask = ~np.isfinite(lam) | (lam < 0.0)
        good = ~miss_mask
        s_theta = np.zeros(N, dtype=np.int8)
        if good.any():
            pol = self.polar_anchor()
            s0 = self.s_theta0.astype(np.float64)
            u_good = (pol['u_obs'][good]
                       + s0[good] * lam[good] * pol['sqrt_udiff'][good])
            sn_good, _, _, _ = ellipj(u_good, pol['k2'][good])
            sgn_good = np.sign(sn_good).astype(np.int8)
            # At exact turning point sn = 0 — fall back to s_theta0 sign.
            sgn_good = np.where(sgn_good == 0,
                                  self.s_theta0[good],
                                  sgn_good).astype(np.int8)
            s_theta[good] = (self.s_theta0[good] * sgn_good).astype(np.int8)
        return s_theta

    # -------------------- photon 4-momentum helpers -----------------------

    def p_at_observer(self):
        """Contravariant photon 4-momentum p^μ at the observer event,
        in the FORWARD-PHYSICAL convention (the photon's actual
        direction of motion in physical time).

        Returns
        -------
        p : (N, 4) array — (p^t, p^r, p^θ, p^φ) in BL, one row per ray.

        The normalization is E = −p_t = +1 (affine choice).  For a
        forward-emission batch (`trace_direction = +1`) this is just
        the Mino-time `photon_p_contravariant`; for a backward-trace
        batch (`trace_direction = −1`) the spatial components (p^r,
        p^θ) are flipped to bring them in line with the physical
        photon direction.  The temporal and azimuthal components are
        invariant since they depend only on (E, L_z, r, θ).

        Non-hit geodesics never have a special observer-event; all
        rays are live at the observer, so there are no NaN rows.
        """
        r = np.full(self.N, self.r0, dtype=np.float64)
        th = np.full(self.N, self.theta0, dtype=np.float64)
        p = photon_p_contravariant(self.a, self.xi, self.eta, r, th,
                                    self.s_r0, self.s_theta0)
        if self.trace_direction != +1:
            # Mino → forward-physical: flip spatial p, keep p^t, p^φ.
            p = p * np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return p

    def p_at_hit(self, event):
        """Contravariant photon 4-momentum p^μ at the hit event, in
        the FORWARD-PHYSICAL convention (the photon's actual direction
        of motion in physical time).

        Parameters
        ----------
        event : dict
            Output from `hit_event` (or anything with at least
            `r_hit`, `theta_hit`, `s_r_at_hit` arrays, plus either
            `s_theta_at_hit` or `lam_hit` so that s_θ at the hit can be
            recovered).  ``s_r_at_hit`` and ``s_theta_at_hit`` carried
            in `event` are MINO-time signs (the natural output of the
            geodesic integrator); the Mino → forward-physical
            conversion is applied here via `self.trace_direction`.

        Returns
        -------
        p : (N, 4) array — (p^t, p^r, p^θ, p^φ) in BL, one row per ray.
            Non-hit rays carry NaN in every component.
        """
        r_hit = np.asarray(event['r_hit'], dtype=np.float64)
        theta_hit = np.asarray(event['theta_hit'], dtype=np.float64)
        s_r_at_hit = np.asarray(event['s_r_at_hit'], dtype=np.float64)
        if 's_theta_at_hit' in event:
            s_theta_at_hit = np.asarray(event['s_theta_at_hit'], dtype=np.float64)
        else:
            lam = np.asarray(event['lam_hit'], dtype=np.float64)
            s_theta_at_hit = self._s_theta_at(lam).astype(np.float64)
        miss = ~np.isfinite(r_hit) | ~np.isfinite(theta_hit)
        # Avoid feeding NaNs into the momentum formula; fill with dummies
        # and NaN the result instead (keeps the elliptic routines happy).
        r_safe = np.where(miss, 2.0, r_hit)
        th_safe = np.where(miss, 0.5 * np.pi, theta_hit)
        sr_safe = np.where(miss, 1.0, s_r_at_hit)
        sth_safe = np.where(miss, 1.0, s_theta_at_hit)
        p = photon_p_contravariant(self.a, self.xi, self.eta,
                                    r_safe, th_safe, sr_safe, sth_safe)
        if self.trace_direction != +1:
            # Mino → forward-physical: flip spatial p, keep p^t, p^φ.
            p = p * np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        # NaN out the miss rays across all four components
        if miss.any():
            p = np.where(miss[:, None], np.nan, p)
        return p
