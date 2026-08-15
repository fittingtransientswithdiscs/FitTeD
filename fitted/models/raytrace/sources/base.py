"""Source — abstract emitter for polarized / redshift pipelines.

A `Source` supplies, at each BL emission event (r, θ):

  * u^μ : 4-velocity of the emitter (contravariant, normalized such that
           g_{μν} u^μ u^ν = −1).
  * n_disk^μ : 4-vector orthogonal to u^μ that encodes the emission
           surface normal (used by scattering-atmosphere emission
           models).  For a thin disk on θ = π/2 this is the unit
           spatial vector along +ẑ in the emitter rest frame.
  * valid : boolean — whether the event is inside the emission region
           at all.

This is the "inverse" of the Observer abstraction: observers say
*where the ray starts* given a pixel; sources say *where the ray
terminates* given a BL event, with the physical 4-velocity and disk
geometry needed to interpret the emission.

Source is parallel to (and eventually the bridge to) the redshift
pipeline: g = (u · p)_em / (u · p)_obs requires u_em^μ from a Source.
"""

from abc import ABC, abstractmethod


class Source(ABC):
    """Abstract base class for an emitter.

    Concrete subclasses implement `u_of(r, theta)` and
    `n_disk_of(r, theta)`.  All methods should broadcast over arrays
    of (r, θ) and return arrays of matching leading shape.
    """

    @abstractmethod
    def u_of(self, r, theta):
        """Emitter 4-velocity u^μ(r, θ), contravariant BL components.

        Returns shape (..., 4).  Events outside the emission region
        should return NaN in all components.
        """

    @abstractmethod
    def n_disk_of(self, r, theta):
        """Emitter-frame disk-normal 4-vector (spatial, unit, orthogonal
        to u^μ).

        For a thin disk on θ = π/2, this is the boosted form of +ẑ in
        the rest frame.  Returns shape (..., 4).
        """

    def valid_mask(self, r, theta):
        """Boolean mask identifying valid emission events."""
        import numpy as np
        u = self.u_of(r, theta)
        return np.isfinite(u[..., 0])
