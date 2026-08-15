"""KerrDisk — equatorial-geodesic fluid on the full r > r_+ range.

This is the natural "thin disk" fluid in Kerr: the covariant energy
E = −u_t and angular momentum L = u_φ are CONSERVED along the orbit
in both branches, and the two branches match continuously at r_ISCO.

Two branches of equatorial motion:

  * r ≥ r_ISCO : circular geodesic (Keplerian) with u^r = 0.
  * r_+ < r < r_ISCO : free-falling plunging geodesic that carries the
    ISCO values (E_ISCO, L_ISCO) inward.  u^r is set by u·u = −1.

The analytic formulas (Mummery, Thorne_Limit/thorne_limit.py, cf.
Page-Thorne 1974) are:

    u_t   = −E_em = −(1 − 2/r + a·r^{−3/2}) / √(1 − 3/r + 2a·r^{−3/2})
                                                       [r ≥ r_ISCO]
    u_φ   = +L_em = √r · (1 + a²/r² − 2a·r^{−3/2}) / √(...)
                                                       [r ≥ r_ISCO]
    u_t   = −√(1 − 2/(3 r_ISCO))                       [r < r_ISCO]
    u_φ   = 2√3 · (1 − 2a/(3 √r_ISCO))                 [r < r_ISCO]
    u^r   = −√(2/(3 r_ISCO)) · (r_ISCO/r − 1)^{3/2}    [r < r_ISCO]
    u^r   = 0                                          [r ≥ r_ISCO]
    u^θ   = 0                                          [both]

The contravariant (u^t, u^φ) components are reconstructed from the
equatorial inverse-metric identities:

    u^t   = (−(r³ + r a² + 2 a²) u_t − 2 a u_φ) / ( r Δ )
    u^φ   = (−2 a u_t + (r − 2) u_φ) / ( r Δ ),

with Δ = r² − 2 r + a².

Note: the class name "NovikovThorne" was reserved for the thin-disk
EMISSION model (Novikov & Thorne 1973), which by construction
truncates at r_ISCO (the classic zero-torque-at-ISCO boundary
condition).  The FLUID KINEMATICS we encode here are in fact
well-defined on both sides of r_ISCO — the plunging continuation is
the analytic extension of the Keplerian orbit that an orbiting
particle actually follows.  Calling it "KerrDisk" makes this clear
and keeps the class available for modelling hot-gas emission
*inside* the ISCO.

Retrograde handling
-------------------
Retrograde fluid around a BH of spin +|a| is obtained by the formal
substitution a → −a in the prograde formulas above, with an overall
sign flip on u_φ (to get the retrograde angular momentum in the +|a|
spacetime).  This is a mathematical identity: the spacetime with
signed spin a is related to the +|a| spacetime by φ → −φ, which
flips the sign of a in the metric and the sign of u^φ in every
trajectory.  So "retrograde orbit in +|a|" ≡ "prograde orbit in −|a|
with φ-axis reversed".  In our KerrDisk(a, prograde=False) the
u^μ output is in the user's +|a| BL coordinates directly.
"""

import numpy as np

from .base import Source
from .keplerian import _isco_radius


class KerrDisk(Source):
    """Equatorial Kerr geodesic fluid on r > r_+, both sides of r_ISCO.

    Parameters
    ----------
    a : float
        BH dimensionless spin (0 ≤ a < 1 recommended; negative a is
        permitted but `prograde` flag is then interpreted relative to
        the sign of a).
    prograde : bool, default True
        Corotation sense with the BH spin.  For a > 0:
            prograde=True  → u^φ > 0 (co-rotating)
            prograde=False → u^φ < 0 (counter-rotating)
    epsilon_theta : float, default 1e-6
        Tolerance on |θ − π/2|.  Events farther than this from the
        equator are treated as off-disk (u^μ NaN).
    """

    def __init__(self, a, prograde=True, epsilon_theta=1e-6):
        self.a = float(a)
        self.prograde = bool(prograde)
        self.epsilon_theta = float(epsilon_theta)
        # Sign convention: sign=+1 prograde, -1 retrograde.  The
        # prograde-formula "effective a" is sign · a; the overall sign
        # of u_φ in the user's +|a| spacetime is also given by `sign`.
        self._sign = +1.0 if self.prograde else -1.0
        self._a_eff = self._sign * self.a
        # ISCO in the user's +|a| spacetime, selecting prograde or
        # retrograde branch appropriately (BPT formula).
        self.r_isco = _isco_radius(self.a, self.prograde)

    # -------- covariant (u_t, u_φ) helpers -----------------------------

    def _u_cov_outside(self, r):
        """Covariant u_t, u_φ at r ≥ r_ISCO (equatorial circular orbit).

        Uses Mummery's prograde formulas with the effective spin
        a_eff = sign · a, then flips u_φ by `sign` to land in the
        user's +|a| spacetime.  For sign=+1 (prograde) this is a
        no-op; for sign=−1 (retrograde) the u_φ returned is negative,
        as expected.
        """
        a_eff = self._a_eff
        r_32 = r ** (-1.5)
        # FitTeD vendored copy: inside the marginally-stable orbit the
        # radicand goes negative and denom becomes NaN.  Those pixels are
        # masked out downstream by `valid`, so the NaN is intentional --
        # errstate suppresses the RuntimeWarning without altering any value.
        with np.errstate(invalid='ignore', divide='ignore'):
            denom = np.sqrt(1.0 - 3.0 / r + 2.0 * a_eff * r_32)
            u_t = -(1.0 - 2.0 / r + a_eff * r_32) / denom
            u_phi_mag = np.sqrt(r) * (
                1.0 + a_eff * a_eff / (r * r) - 2.0 * a_eff * r_32
            ) / denom
        u_phi = self._sign * u_phi_mag
        return u_t, u_phi

    def _u_isco_vals(self):
        """Conserved (u_t, u_φ) at r = r_ISCO in user's +|a| spacetime."""
        return self._u_cov_outside(np.asarray(self.r_isco))

    # -------- contravariant u^μ on the equator --------------------------

    def _u_of_equatorial(self, r):
        """Return contravariant (u^t, u^r, u^θ, u^φ) on the equator at BL r.

        u_t, u_φ come from _u_cov_outside (r ≥ r_ISCO) or are the
        conserved ISCO constants (r < r_ISCO).  u^r comes from the
        plunge formula (r < r_ISCO) or is zero (r ≥ r_ISCO).
        Then we contract against the equatorial BL inverse metric
        (with the user's true a) to get the contravariant form.
        """
        a = self.a
        r = np.asarray(r, dtype=np.float64)
        r_isco = self.r_isco
        r_plus = 1.0 + np.sqrt(max(1.0 - a * a, 0.0))
        Delta = r * r - 2.0 * r + a * a
        in_plunge = (r < r_isco)
        out_plunge = ~in_plunge
        on_geom = (r > r_plus + 1e-12)

        # Branch 1: r ≥ r_ISCO — Keplerian covariant, u^r = 0
        u_t_out = np.zeros_like(r)
        u_phi_out = np.zeros_like(r)
        if out_plunge.any():
            u_t_v, u_phi_v = self._u_cov_outside(r[out_plunge])
            u_t_out[out_plunge] = u_t_v
            u_phi_out[out_plunge] = u_phi_v

        # Branch 2: r_+ < r < r_ISCO — conserved ISCO constants
        u_t_in, u_phi_in = self._u_isco_vals()   # scalars
        u_t_in = float(u_t_in); u_phi_in = float(u_phi_in)

        u_t_cov = np.where(in_plunge, u_t_in, u_t_out)
        u_phi_cov = np.where(in_plunge, u_phi_in, u_phi_out)

        # Equatorial BL inverse metric (note: we use the USER's a, not
        # a_eff, because the spacetime is the user's +|a| one).
        with np.errstate(invalid='ignore', divide='ignore'):
            r_denom = r * Delta
            u_t_con = (-(r ** 3 + r * a * a + 2.0 * a * a) * u_t_cov
                        - 2.0 * a * u_phi_cov) / r_denom
            u_phi_con = (-2.0 * a * u_t_cov
                          + (r - 2.0) * u_phi_cov) / r_denom

        # u^r: zero outside ISCO, plunge formula inside (a-independent,
        # so identical for prograde and retrograde).
        u_r_con = np.where(
            in_plunge & on_geom,
            -np.sqrt(2.0 / (3.0 * r_isco))
              * np.power(np.maximum(r_isco / np.where(r > 0, r, 1.0) - 1.0,
                                     0.0), 1.5),
            0.0,
        )
        u_theta_con = np.zeros_like(r)

        # Zero everything that's not on_geom.
        u_t_con = np.where(on_geom, u_t_con, np.nan)
        u_phi_con = np.where(on_geom, u_phi_con, np.nan)
        u_r_con = np.where(on_geom, u_r_con, np.nan)
        u_theta_con = np.where(on_geom, u_theta_con, np.nan)

        return u_t_con, u_r_con, u_theta_con, u_phi_con

    # -------- Source API -----------------------------------------------

    def u_of(self, r, theta):
        """Contravariant u^μ at (r, θ) for the equatorial Kerr fluid.

        Returns NaN for |θ − π/2| > epsilon_theta or r ≤ r_+.
        """
        r = np.asarray(r, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        r, theta = np.broadcast_arrays(r, theta)
        r_plus = 1.0 + np.sqrt(max(1.0 - self.a * self.a, 0.0))
        on_disk = (np.abs(theta - 0.5 * np.pi) < self.epsilon_theta) & \
                  (r > r_plus + 1e-12)
        r_safe = np.where(on_disk, r, 10.0)

        u_t_con, u_r_con, u_theta_con, u_phi_con = self._u_of_equatorial(r_safe)

        shape = r.shape
        u = np.full(shape + (4,), np.nan, dtype=np.float64)
        u[..., 0] = np.where(on_disk, u_t_con, np.nan)
        u[..., 1] = np.where(on_disk, u_r_con, np.nan)
        u[..., 2] = np.where(on_disk, u_theta_con, np.nan)
        u[..., 3] = np.where(on_disk, u_phi_con, np.nan)
        return u

    def n_disk_of(self, r, theta):
        """Disk normal in the emitter rest frame.

        Since u^θ = 0 in both branches, ∂_θ is already orthogonal to
        u at the equator and the rest-frame disk normal is simply
        −∂_θ / √g_θθ = −∂_θ / r.

        Note: inside r_ISCO the fluid has u^r ≠ 0, so building a
        full emitter-rest-frame tetrad requires a ZAMO-frame detour
        (see kerrgeo.tetrad.emitter_tetrad) — but the disk normal
        itself is unchanged because u · ∂_θ = 0.
        """
        r = np.asarray(r, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        r, theta = np.broadcast_arrays(r, theta)
        r_plus = 1.0 + np.sqrt(max(1.0 - self.a * self.a, 0.0))
        on_disk = (np.abs(theta - 0.5 * np.pi) < self.epsilon_theta) & \
                  (r > r_plus + 1e-12)
        shape = r.shape
        n = np.full(shape + (4,), np.nan, dtype=np.float64)
        n[..., 0] = np.where(on_disk, 0.0, np.nan)
        n[..., 1] = np.where(on_disk, 0.0, np.nan)
        n[..., 2] = np.where(on_disk, -1.0 / r, np.nan)
        n[..., 3] = np.where(on_disk, 0.0, np.nan)
        return n
