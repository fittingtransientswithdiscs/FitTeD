"""AsymptoticObserver — the classical Bardeen camera at r → ∞.

In practice we use a finite r_obs (default 10³ for raw, 10⁵ for the
Gralla-Lupsasca regularised variant).  Concrete Bardeen map:

    ξ  = −α sin i
    η  =  β² + (α² − a²) cos² i
    μ_obs = cos i
    s_θ₀  = −sign(β)   (sign of dμ/dλ at the observer; convention from bardeen.py)

Per-pixel radial turning-point structure (r_hit, s_r_disk, fate code) is
precomputed via `camera_hit_radius` and carried alongside each geodesic
as auxiliary metadata; the target layer reads r_hit / s_r_disk from the
per-ray metadata.

This is the observer that reproduces `camera_tphi` bit-for-bit.
"""

import numpy as np

from ..bardeen import bardeen_transform
from ..camera_pipeline import camera_hit_radius, PIX_HIT

from .base import Observer
# NOTE (FitTeD vendored copy): the scalar geodesic classes are only used by
# make_geodesics(), which the SED path never calls.  They are imported lazily
# so that the (t,phi) elliptic-integral stack is not a dependency of the
# vectorised make_batch() path.  See raytrace/PROVENANCE.md.
from ..geodesic_batch import PhotonGeodesicBatch


class AsymptoticObserver(Observer):
    """Observer at r = r_obs, inclination i, looking back along the axis.

    Parameters
    ----------
    r_obs       : float
        Observer radius in M units.  Default 1000.
    inclination : float
        Inclination in radians from the spin axis, i ∈ (0, π).

    Notes
    -----
    The initial event stored on each geodesic has (t₀, φ₀) = (0, 0).
    The geodesic's equator-hit then returns absolute (t_hit, φ_hit)
    referenced to that zero, matching the conventions of camera_tphi.
    """

    def __init__(self, r_obs, inclination):
        self.r_obs = float(r_obs)
        self.inclination = float(inclination)

    # -------------------------------------------------------------------

    def _preflight(self, a, pixels):
        """Run the Bardeen + radial-turning-point precomputations shared
        across geodesic construction and extra_diagnostics."""
        alpha = np.atleast_1d(np.asarray(pixels["alpha"], dtype=np.float64))
        beta = np.atleast_1d(np.asarray(pixels["beta"], dtype=np.float64))
        shape = np.broadcast_shapes(alpha.shape, beta.shape)
        alpha_flat = np.broadcast_to(alpha, shape).ravel()
        beta_flat = np.broadcast_to(beta, shape).ravel()

        # Bardeen conserved quantities + observer sign.
        xi, eta, mu_obs, s_mu_obs = bardeen_transform(
            a, self.inclination, alpha_flat, beta_flat,
        )
        xi = np.asarray(xi).ravel()
        eta = np.asarray(eta).ravel()
        mu_obs = np.asarray(mu_obs).ravel()
        s_mu_obs = np.asarray(s_mu_obs).ravel()

        # Radial fate: r_hit, pixel code, s_r_disk (disk-crossing radial sign),
        # and lam_hit (forward Mino time at the equator crossing — free
        # by-product of the polar elliptic work that already computed r_hit).
        r_hit, code, s_r_disk, lam_hit = camera_hit_radius(
            a, self.inclination, alpha_flat, beta_flat,
            return_sr=True, return_lam=True,
        )
        r_hit = np.asarray(r_hit).ravel()
        code = np.asarray(code).ravel()
        s_r_disk = np.asarray(s_r_disk).ravel()
        lam_hit = np.asarray(lam_hit).ravel()

        return dict(
            alpha=alpha_flat, beta=beta_flat,
            xi=xi, eta=eta,
            mu_obs=mu_obs, s_mu_obs=s_mu_obs,
            r_hit=r_hit, code=code, s_r_disk=s_r_disk,
            lam_hit=lam_hit,
            shape=shape,
        )

    # -------------------------------------------------------------------

    def make_geodesics(self, a, pixels):
        from ..geodesic import KerrPhotonGeodesic, SchwarzschildPhotonGeodesic  # lazy
        """Construct one PhotonGeodesic per pixel.

        Non-HIT pixels (captured, escaped, etc.) still get a geodesic
        object so that the ordering is preserved; callers skip them by
        consulting extra_diagnostics()['code'].
        """
        a = float(a)
        pre = self._preflight(a, pixels)
        N = pre["alpha"].size
        schwarzschild = abs(a) < 1e-15

        theta0 = self.inclination
        r0 = self.r_obs

        geodesics = []
        for k in range(N):
            xi_k = float(pre["xi"][k])
            eta_k = float(pre["eta"][k])
            s_mu_k = float(pre["s_mu_obs"][k])
            # The initial sign of dr/dλ at an asymptotic observer looking
            # inward is always s_r0 = −1 (inward-moving forward photon).
            s_r0 = -1

            if schwarzschild:
                g = SchwarzschildPhotonGeodesic(
                    a=0.0,
                    xi=xi_k, eta=eta_k,
                    r0=r0, theta0=theta0, t0=0.0, phi0=0.0,
                    s_r0=s_r0, s_theta0=int(np.sign(s_mu_k)) if s_mu_k != 0 else +1,
                    camera_metadata=dict(
                        alpha=float(pre["alpha"][k]),
                        beta=float(pre["beta"][k]),
                        i_rad=self.inclination,
                    ),
                )
            else:
                g = KerrPhotonGeodesic(
                    a=a,
                    xi=xi_k, eta=eta_k,
                    r0=r0, theta0=theta0, t0=0.0, phi0=0.0,
                    s_r0=s_r0, s_theta0=int(np.sign(s_mu_k)) if s_mu_k != 0 else +1,
                )
            geodesics.append(g)

        return geodesics

    # -------------------------------------------------------------------

    def make_batch(self, a, pixels):
        """Construct a single PhotonGeodesicBatch for the whole image.

        Returns
        -------
        batch : PhotonGeodesicBatch
            Shared (r₀, θ₀) = (r_obs, inclination); per-ray (ξ, η,
            s_r0, s_θ0) arrays.  All rays start inward (s_r0 = −1).
        ray_metadata : dict
            Per-ray arrays ready to feed `Target.first_hit_batch`:
            ``code``, ``r_hit``, ``s_r_disk``, plus the scalar
            ``PIX_HIT`` enum.  Mirrors the per-pixel dict used in the
            scalar path, but with arrays of length N.
        shape : tuple
            Original broadcast shape of (α, β); callers reshape
            outputs via `event[k].reshape(shape)`.
        """
        a = float(a)
        if abs(a) < 1e-15:
            raise NotImplementedError(
                "Vectorised (Phase 4b) AsymptoticObserver.make_batch does not "
                "yet dispatch to the Schwarzschild backend; use "
                "make_geodesics() for a = 0 science cases."
            )
        pre = self._preflight(a, pixels)
        N = pre["alpha"].size
        s_mu = np.where(pre["s_mu_obs"] == 0, +1.0, pre["s_mu_obs"])
        s_theta0 = np.sign(s_mu).astype(np.int8)
        s_r0 = -np.ones(N, dtype=np.int8)  # asymptotic observer: inward

        batch = PhotonGeodesicBatch(
            a=a,
            xi=pre["xi"], eta=pre["eta"],
            r0=self.r_obs, theta0=self.inclination,
            t0=0.0, phi0=0.0,
            s_r0=s_r0, s_theta0=s_theta0,
            trace_direction=-1,   # backward observer trace: λ runs
                                   # opposite to physical photon time
        )
        ray_metadata = dict(
            code=pre["code"].astype(np.int64),
            r_hit=pre["r_hit"],
            s_r_disk=pre["s_r_disk"],
            lam_hit=pre["lam_hit"],
            xi=pre["xi"],
            eta=pre["eta"],
            mu_obs=pre["mu_obs"],
            s_mu_obs=pre["s_mu_obs"],
            alpha=pre["alpha"],
            beta=pre["beta"],
            PIX_HIT=PIX_HIT,
        )
        return batch, ray_metadata, pre["shape"]

    # -------------------------------------------------------------------

    def extra_diagnostics(self, a, pixels):
        """Per-ray metadata that targets and users may need:

            code       : int array, pixel fate (PIX_HIT etc.)
            r_hit      : float array, equatorial-crossing radius
            s_r_disk   : ±1 array, radial sign at the disk crossing
            xi, eta    : conserved quantities (diagnostic)
            mu_obs, s_mu_obs : observer polar data (diagnostic)
            shape      : original broadcast shape of (α, β)
        """
        pre = self._preflight(a, pixels)
        return dict(
            code=pre["code"].astype(int),
            r_hit=pre["r_hit"],
            s_r_disk=pre["s_r_disk"],
            xi=pre["xi"],
            eta=pre["eta"],
            mu_obs=pre["mu_obs"],
            s_mu_obs=pre["s_mu_obs"],
            alpha=pre["alpha"],
            beta=pre["beta"],
            shape=pre["shape"],
            PIX_HIT=PIX_HIT,
        )
