"""KeplerianDisk — prograde equatorial circular-geodesic emitter.

Supplies the 4-velocity of a test-fluid element on a circular geodesic
in the equatorial plane (θ = π/2) at radius r, for r ≥ r_ISCO.

Formulas: Bardeen–Press–Teukolsky 1972 for the ISCO radius; Thorne
(1974) / Novikov–Thorne for the circular-geodesic 4-velocity.  In
M = 1 units with prograde (+φ) orbital sense:

  Ω(r)     = 1 / (r^{3/2} + a)
  u^t(r)   = 1 / √(1 − 2/r + 4 a Ω / r − Ω² (r² + a² + 2 a² / r))
  u^φ(r)   = Ω · u^t
  u^r, u^θ = 0

The denominator reduces to (Δ · (r² − 3 r + 2 a √r)) / r² up to
standard manipulation; we use the direct form above because it is
numerically stable without intermediate cancellation.

Outside the equator (θ ≠ π/2) the source returns NaN.  Inside r_ISCO,
a proper plunging-geodesic branch would switch to a non-circular
4-velocity; for the first pass we NaN the source there too (the
plunging region is roadmap deferred work).

Disk normal n_disk^μ : in the emitter rest frame this is +ẑ (the
spin-axis direction).  Taking the BL 1-form dθ (which points in the
+θ direction = AWAY from the spin axis on 0 < θ < π/2), we have
n_disk_contravariant = −∂_θ / √g_θθ  at the equator, boosted to u^μ
rest frame.  Because ∂_θ is orthogonal to ∂_t and ∂_φ and u has no
θ-component, the rest-frame spatial direction is simply the BL
−θ direction, unit-normalised by g_θθ = Σ = r².
"""

import numpy as np

from .base import Source


def _isco_radius(a, prograde=True):
    """Bardeen–Press–Teukolsky ISCO radius.

    Parameters
    ----------
    a : float, |a| ≤ 1
    prograde : bool, default True
        If True, compute prograde (corotating, smaller) ISCO; if False
        compute retrograde (counter-rotating, larger) ISCO.
    """
    a = float(a)
    if abs(a) > 1.0:
        raise ValueError(f"|a| must be ≤ 1; got a = {a}")
    Z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0)
    )
    Z2 = np.sqrt(3.0 * a * a + Z1 * Z1)
    sign = -1.0 if prograde else +1.0
    return 3.0 + Z2 + sign * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))


class KeplerianDisk(Source):
    """Thin prograde Keplerian disk on θ = π/2 outside r_ISCO.

    Parameters
    ----------
    a : float — BH spin (must match the spin of the geodesic tracer).
    prograde : bool, default True.
    epsilon_theta : float, default 1e-6
        Tolerance on |θ − π/2| for "on the equator".  Events farther
        away are treated as misses and u^μ is NaN.
    """

    def __init__(self, a, prograde=True, epsilon_theta=1e-6):
        self.a = float(a)
        self.prograde = bool(prograde)
        self.epsilon_theta = float(epsilon_theta)
        self.r_isco = _isco_radius(self.a, self.prograde)

    def u_of(self, r, theta):
        r = np.asarray(r, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        r, theta = np.broadcast_arrays(r, theta)
        a = self.a
        sign = +1.0 if self.prograde else -1.0

        on_disk = (np.abs(theta - 0.5 * np.pi) < self.epsilon_theta) & \
                  (r >= self.r_isco)

        # Safe eval of Keplerian quantities (everything NaN off-disk).
        r_safe = np.where(on_disk, r, 10.0)
        Omega = sign / (r_safe ** 1.5 + a * sign)
        # u^t from −g_tt − 2 Ω g_tφ − Ω² g_φφ at θ = π/2:
        #   −g_tt    = 1 − 2/r
        #   −2 Ω g_tφ = 4 a Ω / r
        #   −Ω² g_φφ = −Ω² (r² + a² + 2 a² / r)
        denom = (1.0 - 2.0 / r_safe + 4.0 * a * Omega / r_safe
                  - Omega * Omega * (r_safe * r_safe + a * a
                                      + 2.0 * a * a / r_safe))
        # Numerical floor
        denom = np.where(denom > 0.0, denom, np.nan)
        u_t = 1.0 / np.sqrt(denom)
        u_phi = Omega * u_t

        shape = r.shape
        u = np.full(shape + (4,), np.nan, dtype=np.float64)
        u[..., 0] = np.where(on_disk, u_t, np.nan)
        u[..., 1] = np.where(on_disk, 0.0, np.nan)
        u[..., 2] = np.where(on_disk, 0.0, np.nan)
        u[..., 3] = np.where(on_disk, u_phi, np.nan)
        return u

    def n_disk_of(self, r, theta):
        """Spatial unit vector along the disk normal (−θ̂ in BL), in the
        emitter rest frame.

        Since u^μ has no θ-component, ∂_θ is already orthogonal to u,
        so n_disk is simply the unit-normalised (−∂_θ) in BL:
            n^μ = (0, 0, −1/√g_θθ, 0)  at θ = π/2, Σ = r².
        """
        r = np.asarray(r, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        r, theta = np.broadcast_arrays(r, theta)
        on_disk = (np.abs(theta - 0.5 * np.pi) < self.epsilon_theta) & \
                  (r >= self.r_isco)
        shape = r.shape
        n = np.full(shape + (4,), np.nan, dtype=np.float64)
        # g_θθ = Σ = r² + a² cos²θ = r² at equator
        n[..., 0] = np.where(on_disk, 0.0, np.nan)
        n[..., 1] = np.where(on_disk, 0.0, np.nan)
        n[..., 2] = np.where(on_disk, -1.0 / r, np.nan)
        n[..., 3] = np.where(on_disk, 0.0, np.nan)
        return n
