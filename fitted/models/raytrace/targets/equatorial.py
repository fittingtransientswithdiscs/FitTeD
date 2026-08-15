"""EquatorialDisk — stopping condition θ = π/2, optionally r ∈ [r_in, r_out].

Phase 1 behaviour: the observer precomputes (r_hit, s_r_disk, fate code)
via `camera_hit_radius` and passes them through `ray_metadata`.
EquatorialDisk reads those and calls the geodesic's `equator_hit`
primitive.  Non-HIT pixels return None.

Phase 2 will switch EquatorialDisk to invert `lam_of_theta(π/2)` on the
geodesic itself and then evaluate r(λ) at that Mino time, closing the
dependency loop with `camera_hit_radius`.
"""

import numpy as np

from .base import Target


class EquatorialDisk(Target):
    """Stopping surface θ = π/2 with optional radial gate [r_in, r_out].

    Parameters
    ----------
    r_in  : float or None
        Inner radius.  A geodesic hit at r < r_in is treated as "misses
        the disk" and returns None.  None disables the inner gate
        (default: r_in = r_+ is already enforced by the pipeline via
        `camera_hit_radius`, so None is the usual choice).
    r_out : float or None
        Outer radius.  A geodesic hit at r > r_out returns None.  None
        disables the outer gate (default).
    regularize : bool
        Subtract the asymptotic linear + log pieces of t_hit.  Matches
        the `regularize` flag of camera_tphi.
    """

    def __init__(self, r_in=None, r_out=None, regularize=False):
        self.r_in = None if r_in is None else float(r_in)
        self.r_out = None if r_out is None else float(r_out)
        self.regularize = bool(regularize)

    def first_hit(self, geodesic, ray_metadata=None, include_tphi=True):
        # The asymptotic observer provides r_hit + s_r_disk + fate code.
        # Phase 2 will let the geodesic find these itself.
        if ray_metadata is None:
            raise ValueError(
                "EquatorialDisk Phase 1 requires ray_metadata with "
                "'r_hit', 's_r_disk', 'code', 'PIX_HIT' from the observer."
            )

        code = int(ray_metadata["code"])
        if code != int(ray_metadata["PIX_HIT"]):
            return None   # captured / escaped / no equator crossing

        r_hit = float(ray_metadata["r_hit"])
        if not np.isfinite(r_hit):
            return None

        if self.r_in is not None and r_hit < self.r_in:
            return None
        if self.r_out is not None and r_hit > self.r_out:
            return None

        s_r_disk = int(np.sign(ray_metadata["s_r_disk"]))
        if s_r_disk == 0:
            s_r_disk = +1

        if include_tphi:
            return geodesic.equator_hit(
                r_at_hit=r_hit,
                s_r_disk=s_r_disk,
                regularize=self.regularize,
            )
        return geodesic.equator_hit_rtheta(
            r_at_hit=r_hit,
            s_r_disk=s_r_disk,
            regularize=self.regularize,
        )

    def first_hit_batch(self, batch, ray_metadata=None, include_tphi=True):
        """Vectorised equator hit.

        The observer supplies the per-ray r_hit, s_r_disk, and fate code
        via `ray_metadata`.  Geometry-only (include_tphi=False) honours
        them directly with no elliptic work.  With include_tphi=True we
        additionally invert the polar to find the Mino time to the
        equator and call batch.hit_event for (t, φ).
        """
        if ray_metadata is None:
            raise ValueError(
                "EquatorialDisk.first_hit_batch requires ray_metadata "
                "with per-ray 'r_hit', 'code', 'PIX_HIT' arrays.")
        N = batch.N
        code = np.asarray(ray_metadata['code'], dtype=np.int64)
        pix_hit = int(ray_metadata['PIX_HIT'])
        r_hit = np.asarray(ray_metadata['r_hit'], dtype=np.float64)
        s_r_disk = np.asarray(ray_metadata['s_r_disk'], dtype=np.float64)

        hit_mask = (code == pix_hit) & np.isfinite(r_hit)
        if self.r_in is not None:
            hit_mask &= (r_hit >= self.r_in)
        if self.r_out is not None:
            hit_mask &= (r_hit <= self.r_out)

        r_out_arr = np.where(hit_mask, r_hit, np.nan)
        theta_out = np.where(hit_mask, 0.5 * np.pi, np.nan)
        # s_r_disk from the observer metadata is a DIRECT/BOUNCE flag
        # (+1 = direct leg, −1 = bounced through pericenter en route),
        # NOT the physical sign of dr/dλ at the disk.  The physical
        # sign at the disk is s_r0 × s_r_disk_flag:
        #   direct (s_r_disk_flag=+1): s_r_at_disk = s_r0 (unchanged)
        #   bounce (s_r_disk_flag=−1): s_r_at_disk = −s_r0 (flipped at pericenter)
        # `s_r_at_hit` is the MINO-time sign of dr/dλ at the disc-side
        # endpoint of the trajectory leg, in the same convention as
        # `batch.s_r0` (Mino-time at λ=0).  A radial bounce between
        # observer and disc flips this sign relative to s_r0, no
        # bounce preserves it — encoded in the direct/bounce flag
        # `s_r_disk` ∈ {±1}.
        #
        # The downstream momentum routines `batch.p_at_hit` consume
        # this Mino sign together with `batch.trace_direction` to
        # produce the FORWARD-PHYSICAL p^μ at the disc emission point;
        # see `geodesic_batch.py` and POLARIZATION_CONVENTIONS.md §4.
        s_r_at_hit = np.where(
            hit_mask,
            (np.sign(s_r_disk) * batch.s_r0.astype(np.float64)).astype(np.int8),
            0,
        ).astype(np.int8)

        if not include_tphi:
            nan_arr = np.full(N, np.nan)
            # The observer (e.g. AsymptoticObserver) may have already
            # computed and surfaced the forward Mino time at the equator
            # crossing as a free by-product of the camera's polar-fate
            # work.  When present, expose it here for downstream consumers
            # that prefer to recompute the polar tangent sign from λ
            # (one ``ellipj`` per ray); when absent, fall back to NaN.
            meta_lam = ray_metadata.get('lam_hit', None)
            if meta_lam is not None:
                lam_hit_arr = np.where(hit_mask,
                                        np.asarray(meta_lam, dtype=np.float64),
                                        np.nan)
            else:
                lam_hit_arr = nan_arr.copy()
            # Closed-form polar tangent sign at the *first* equator crossing
            # (n=0).  The polar oscillation has μ = μ_+ cn(u, k²); the equator
            # is the first u with cn(u) = 0, i.e. u = +K (DIRECT branch:
            # s_θ0 = +1, sn(+K) = +1) or u = −K (BOUNCING branch:
            # s_θ0 = −1, sn(−K) = −1).  In both branches
            # s_θ_at_hit = s_θ0 · sign(sn(u)) = +1, so we set it directly
            # and let ``p_at_hit`` consume it without triggering the polar
            # anchor / Jacobi-elliptic evaluation.  For n>0 image orders
            # this would alternate and a different convention would apply,
            # but EquatorialDisk's first-hit semantics are n=0 by
            # construction.
            s_theta_at_hit = np.where(hit_mask, 1, 0).astype(np.int8)
            return dict(
                t_hit=nan_arr.copy(), phi_hit=nan_arr.copy(),
                r_hit=r_out_arr, theta_hit=theta_out,
                lam_hit=lam_hit_arr,
                tau_hit=nan_arr.copy(), sigma_hit=nan_arr.copy(),
                G_tau=nan_arr.copy(), G_sig=nan_arr.copy(),
                J_tau=nan_arr.copy(), J_sig=nan_arr.copy(),
                s_r_at_hit=s_r_at_hit,
                s_theta_at_hit=s_theta_at_hit,
            )

        # Full (t, φ) path: invert polar to get Mino time to equator,
        # then evaluate hit_event (which computes G_τ, G_σ, J_τ, J_σ
        # and assembles t, φ via the PD split).
        lam_pol = batch.lam_of_theta(0.5 * np.pi, n_crossing=0)
        # Rays that don't hit the disk (per the observer's fate code) get
        # NaN lam so hit_event sees them as miss and returns NaN.
        lam_masked = np.where(hit_mask, lam_pol, np.nan)
        # Feed the known r_hit as r_at_lam to avoid an extra r_of_lam call;
        # for n=0 equator crossing this is the authoritative observer
        # value from camera_hit_radius metadata.
        event = batch.hit_event(lam_masked,
                                 r_at_lam=r_out_arr,
                                 s_r_at_lam=s_r_at_hit.astype(np.int8),
                                 regularize=self.regularize)
        # Pin theta_hit to π/2 on hit pixels (small numerical noise
        # from arccos(clip(mu_+·cn, ...)) otherwise).
        event['theta_hit'] = theta_out
        event['r_hit'] = r_out_arr
        return event
