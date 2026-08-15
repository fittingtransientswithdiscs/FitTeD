"""Target abstract base class.

A Target is a stopping-condition hypersurface in Kerr coordinates.
Subclasses implement:

    first_hit(geodesic, ray_metadata=None) -> dict or None

which returns the BL hit event (t_hit, r_hit, θ_hit, φ_hit, λ_hit) and
any per-target diagnostics, or None if the geodesic does not reach the
target (captured, escaped, misses the cone, etc.).

`ray_metadata` carries optional per-ray auxiliary data that the target
may need (e.g. the precomputed r_hit / s_r_disk / fate code from
`AsymptoticObserver.extra_diagnostics`).  Targets that don't need it
ignore it.
"""

from abc import ABC, abstractmethod


class Target(ABC):
    """Abstract stopping hypersurface."""

    @abstractmethod
    def first_hit(self, geodesic, ray_metadata=None, include_tphi=True):
        """Solve for the first event at which `geodesic` hits this target.

        Parameters
        ----------
        geodesic : PhotonGeodesic
        ray_metadata : dict or None
            Observer-provided per-ray diagnostics (e.g. r_hit, s_r_disk,
            fate code from AsymptoticObserver).
        include_tphi : bool, default True
            If False, skip the closed-form (t, φ) assembly (G_τ, G_σ,
            J_τ, J_σ).  The returned event dict carries t_hit=NaN and
            phi_hit=NaN; r_hit and theta_hit are unaffected.  Use for
            purely geometric science cases.

        Returns
        -------
        dict or None
            The hit-event dict as returned by the geodesic's specific
            solver (keys t_hit, phi_hit, r_hit, theta_hit, lam_hit, …),
            or None if the geodesic does not reach the target.
        """
