"""Observer abstract base class.

Concrete observers must implement `make_geodesics(a, pixels)`, which
receives a pixel / sky-direction specification (the exact form is
observer-specific — e.g., AsymptoticObserver expects dict(alpha=, beta=))
and returns a list of PhotonGeodesic instances, one per ray.

Vectorisation note (Phase 1)
----------------------------
Phase 1 returns a Python list of PhotonGeodesic instances so that the
rest of the pipeline can be a straightforward loop matching camera_tphi.
Phase 1.5 / Phase 2 will add a vectorised variant (`make_geodesic_batch`)
that represents an entire pixel batch in a single PhotonGeodesic holding
arrays, sharing the common (a, observer) precomputations.
"""

from abc import ABC, abstractmethod


class Observer(ABC):
    """Abstract camera / observer."""

    @abstractmethod
    def make_geodesics(self, a, pixels):
        """Produce one PhotonGeodesic per pixel.

        Parameters
        ----------
        a      : float
            Kerr spin.
        pixels : observer-specific object
            For AsymptoticObserver, a dict(alpha=..., beta=...) with
            broadcastable arrays.

        Returns
        -------
        list[PhotonGeodesic]
            One geodesic per ray, in flat (row-major) pixel order.
        """

    # Subclasses may override to return additional per-ray metadata
    # (e.g. the `code` array from camera_hit_radius) that callers can
    # thread through to diagnostic dicts.
    def extra_diagnostics(self, a, pixels):
        """Return a dict of per-ray diagnostic arrays (optional)."""
        return {}
