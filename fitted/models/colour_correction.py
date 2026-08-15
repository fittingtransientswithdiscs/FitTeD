"""
Colour-correction prescriptions for FitTeD disc models.

Spectral colour correction `f_col` modifies the locally emitted Planck
spectrum by hardening the apparent temperature:

    B_nu(T_eff) -> (1/f_col**4) * B_nu(f_col * T_eff)

Different prescriptions in the literature express f_col as a function of
the local effective temperature, the local radius, the accretion rate,
or fixed combinations of these.  This module provides:

  * A small ``ColourCorrection`` base class with a uniform call
    signature ``__call__(T, r) -> f_col``, where T is the local
    effective temperature in Kelvin and r is the matching radial array
    in r_g units.  Either argument may be ignored by a prescription
    that does not need it.

  * Built-in concrete prescriptions: ``Done2012``, ``ShimuraTakahara1995``,
    ``Constant``, ``Unity``, ``RadialPiecewise``, ``RadialSmoothPlunge``.

  * A ``REGISTRY`` dict mapping short string names to these classes for
    convenient string-based selection.

  * A ``resolve(spec, *, r_isco=None) -> ColourCorrection`` helper that
    accepts a bool, a str, a number, a callable, or an existing
    ``ColourCorrection`` instance, and returns a ready-to-call instance.

User-facing usage::

    from fitted.models import GR_disc_plus
    from fitted.models import colour_correction as cc

    # Default (Done+ 2012)
    m = GR_disc_plus(colour_correction=True)

    # Pick a built-in by name
    m = GR_disc_plus(colour_correction='shimura_takahara')

    # Constant value (numeric shorthand)
    m = GR_disc_plus(colour_correction=1.7)

    # Radial split
    m = GR_disc_plus(colour_correction=cc.RadialPiecewise(f_disc=1.7,
                                                                f_isco=2.4))

    # Custom callable
    m = GR_disc_plus(colour_correction=lambda T, r: 1.7 + 0.1*np.log10(T/1e7))

The internal call site in the SED engines is ``self.fc(T, r)`` for both
T-only and r-aware prescriptions.  See ``Model_base._set_colour_correction``
for the dispatch logic.

References
----------
* Done, Davis, Jin, Blaes & Ward (2012), MNRAS 420, 1848 — three-segment
  fit f_col(T).  This is the FitTeD default.
* Shimura & Takahara (1995), ApJ 445, 780 — the original log-T form.
* Davis & Hubeny (2005); Davis et al. (2006) — TLUSTY-based atmosphere
  models.  Stub provided as ``Davis2005`` (parameters needed at
  construction time, not fully implemented in this module).
"""

from __future__ import annotations

import copy
import inspect

import numpy as np

__all__ = [
    "ColourCorrection",
    "Done2012", "ShimuraTakahara1995",
    "Constant", "Unity",
    "RadialPiecewise", "RadialPowerLawPlunge", "RadialSmoothPlunge",
    "REGISTRY", "resolve",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ColourCorrection:
    """Abstract base class for f_col prescriptions.

    Concrete subclasses must implement ``__call__(self, T, r) -> ndarray``.
    Either argument may be ignored by prescriptions that don't need it,
    but the signature must accept both for uniform calling.
    """

    def __call__(self, T, r):
        raise NotImplementedError("subclass must implement __call__(T, r)")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# T-only prescriptions
# ---------------------------------------------------------------------------

class Done2012(ColourCorrection):
    """Done, Davis, Jin, Blaes & Ward 2012 three-segment f_col(T).

    f_col = 1                                                  for T <= 3e4 K
    f_col = (T / 3e4)**0.8333598980732597                      for 3e4 < T < 1e5 K
    f_col = (11598 * 72000 / T)**(1/9)                         for T >= 1e5 K

    Reference: Done et al. 2012 MNRAS 420, 1848.
    """

    def __call__(self, T, r):
        T = np.asarray(T, dtype=float)
        i_low  = T <= 3e4
        i_mid  = (T < 1e5) & ~i_low
        i_high = ~(i_low | i_mid)
        f = np.empty_like(T)
        f[i_low]  = 1.0
        f[i_mid]  = (T[i_mid] / 3e4)**0.8333598980732597
        f[i_high] = (11598.0 * 72000.0 / T[i_high])**(1.0 / 9.0)
        return f


class ShimuraTakahara1995(ColourCorrection):
    """Shimura & Takahara 1995 logarithmic f_col(T).

    f_col(T) = 1.7 + 0.05 * log10(T / 3e7 K)

    Reference: Shimura & Takahara 1995, ApJ 445, 780.
    """

    def __call__(self, T, r):
        T = np.asarray(T, dtype=float)
        return 1.7 + 0.05 * np.log10(T / 3e7)


# ---------------------------------------------------------------------------
# Trivial prescriptions
# ---------------------------------------------------------------------------

class Constant(ColourCorrection):
    """Constant f_col = f, independent of T and r.

    Used by the fullkerr paper convention (f = 1.7 throughout) and as a
    quick toggle for sensitivity studies.
    """

    def __init__(self, f: float):
        self.f = float(f)

    def __call__(self, T, r):
        T = np.asarray(T, dtype=float)
        return np.full_like(T, self.f)

    def __repr__(self) -> str:
        return f"Constant(f={self.f})"


class Unity(ColourCorrection):
    """No colour correction (f_col = 1 everywhere)."""

    def __call__(self, T, r):
        T = np.asarray(T, dtype=float)
        return np.ones_like(T)


# ---------------------------------------------------------------------------
# r-aware prescriptions
# ---------------------------------------------------------------------------

class RadialPiecewise(ColourCorrection):
    """Distinct f_col in the main disc and the plunging region.

    f_col(r) = f_disc   for r >= r_isco
             = f_isco   for r <  r_isco

    Mummery, Ingram, Davis & Fabian (2024) use this with f_disc = f_isco
    = 1.7 (i.e. effectively a Constant); this class is here for the
    case where the two regions are deliberately allowed to differ.

    The ``r_isco`` value can be set at construction time, or left as
    ``None`` for the model to inject at SED-prediction time (the model
    knows the spin and hence r_isco).
    """

    def __init__(self, f_disc: float = 1.7, f_isco: float = 1.7,
                 r_isco: float | None = None):
        self.f_disc = float(f_disc)
        self.f_isco = float(f_isco)
        self.r_isco = r_isco

    def __call__(self, T, r):
        if self.r_isco is None:
            raise RuntimeError(
                "RadialPiecewise needs r_isco — supply it at construction "
                "or let the model inject it at SED-prediction time.")
        r = np.asarray(r, dtype=float)
        return np.where(r >= self.r_isco, self.f_disc, self.f_isco)

    def __repr__(self) -> str:
        return (f"RadialPiecewise(f_disc={self.f_disc}, f_isco={self.f_isco}, "
                f"r_isco={self.r_isco})")


class RadialPowerLawPlunge(ColourCorrection):
    """f_col constant outside the ISCO, power-law growth inside the plunge.

        f_col(r) = f_disc                          for r >= r_isco
                 = f_disc · (r_isco/r)**xi          for r <  r_isco

    Continuous (and equal to f_disc) at r = r_isco; grows as a power law
    inward.  This is the prescription used in Mummery, Ingram, Davis &
    Fabian (2024) §6.1 for the MAXI J1820+070 fits, with f_disc = 1.7 and
    xi ≈ 1.7 (best-fit value 1.70166 for the Nu29 epoch).

    Unlike ``RadialSmoothPlunge`` there is no hard cap on f_col deep in
    the plunge — the function grows monotonically as r → r_+.  For a* =
    0.2 (r_+/r_I = 0.37) the deepest pixel reaches f ≈ f_disc · 0.37^(-1.7) ≈ 8.0.
    """

    def __init__(self, f_disc: float = 1.7, xi: float = 1.7,
                 r_isco: float | None = None):
        self.f_disc = float(f_disc)
        self.xi = float(xi)
        self.r_isco = r_isco

    def __call__(self, T, r):
        if self.r_isco is None:
            raise RuntimeError(
                "RadialPowerLawPlunge needs r_isco — supply it at construction "
                "or let the model inject it at SED-prediction time.")
        r = np.asarray(r, dtype=float)
        f = np.full_like(r, self.f_disc)
        inside = r < self.r_isco
        if np.any(inside):
            f[inside] = self.f_disc * (self.r_isco / r[inside])**self.xi
        return f

    def __repr__(self) -> str:
        return (f"RadialPowerLawPlunge(f_disc={self.f_disc}, xi={self.xi}, "
                f"r_isco={self.r_isco})")


class RadialSmoothPlunge(ColourCorrection):
    """f_col grows smoothly inside the plunge.

    f_col(r) = f_disc                        for r >= r_isco
             = min(f_disc·(r_isco/r)^p,
                   f_inner)                  for r <  r_isco

    Captures the heating of the plunging fluid without a discontinuity
    at the ISCO.  Hand-wavy but useful for sensitivity tests.
    """

    def __init__(self, f_disc: float = 1.7, f_inner: float = 2.4,
                 slope: float = 0.4, r_isco: float | None = None):
        self.f_disc = float(f_disc)
        self.f_inner = float(f_inner)
        self.slope = float(slope)
        self.r_isco = r_isco

    def __call__(self, T, r):
        if self.r_isco is None:
            raise RuntimeError(
                "RadialSmoothPlunge needs r_isco — supply it at construction "
                "or let the model inject it at SED-prediction time.")
        r = np.asarray(r, dtype=float)
        f = np.full_like(r, self.f_disc)
        inside = r < self.r_isco
        if np.any(inside):
            f_in = self.f_disc * (self.r_isco / r[inside])**self.slope
            f[inside] = np.minimum(f_in, self.f_inner)
        return f

    def __repr__(self) -> str:
        return (f"RadialSmoothPlunge(f_disc={self.f_disc}, f_inner={self.f_inner}, "
                f"slope={self.slope}, r_isco={self.r_isco})")


# ---------------------------------------------------------------------------
# String registry
# ---------------------------------------------------------------------------

REGISTRY: dict = {
    'done2012': Done2012,
    'done': Done2012,
    'shimura_takahara': ShimuraTakahara1995,
    'st1995': ShimuraTakahara1995,
    'shimura': ShimuraTakahara1995,
    'unity': Unity,
    'none': Unity,
    'off': Unity,
    'radial_piecewise': RadialPiecewise,
    'radial_powerlaw_plunge': RadialPowerLawPlunge,
    'fullkerr': RadialPowerLawPlunge,            # Mummery+ 2024 paper convention
    'radial_smooth_plunge': RadialSmoothPlunge,
    # 'constant' deliberately omitted: needs a parameter, use Constant(f)
    # or pass the numeric value directly.
}


# ---------------------------------------------------------------------------
# Callable adapter for user-supplied (T) or (T, r) functions
# ---------------------------------------------------------------------------

class _CallableWrap(ColourCorrection):
    """Wrap an arbitrary callable into a ColourCorrection.

    Detects whether the callable accepts one positional arg (T) or two
    (T, r) and adapts.  Used internally by ``resolve``.
    """

    def __init__(self, fn):
        self._fn = fn
        try:
            sig = inspect.signature(fn)
            n = sum(1 for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
                    and p.default is p.empty)
            self._takes_r = (n >= 2) or any(
                p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
        except (TypeError, ValueError):
            # Builtins / partials without a signature: assume new (T, r).
            self._takes_r = True

    def __call__(self, T, r):
        return self._fn(T, r) if self._takes_r else self._fn(T)

    def __repr__(self) -> str:
        return f"_CallableWrap({self._fn!r}, takes_r={self._takes_r})"


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def resolve(spec, *, r_isco=None) -> ColourCorrection:
    """Convert any ``colour_correction`` spec into a callable instance.

    Accepted forms:
      * ``True``   -> Done2012()        (legacy default-on)
      * ``False``  -> Unity()           (legacy default-off)
      * ``str``    -> REGISTRY[str]()   ('done2012', 'unity', etc.)
      * ``int|float`` -> Constant(value)
      * ``ColourCorrection`` instance   -> shallow-copy + auto-fill r_isco
      * any callable -> wrapped to (T, r) signature

    The ``r_isco`` keyword is auto-injected into r-aware instances
    (those whose ``r_isco`` attribute exists and is None) so users
    don't have to pass it manually.  A *copy* is made before injection
    so the original instance is unaffected.

    Returns a ColourCorrection (or ColourCorrection-like) instance ready
    to be called as ``inst(T, r)``.
    """
    if spec is True:
        return Done2012()
    if spec is False:
        return Unity()
    if isinstance(spec, ColourCorrection):
        # Don't mutate the user's instance: copy then inject if needed.
        if hasattr(spec, 'r_isco') and getattr(spec, 'r_isco', None) is None \
                and r_isco is not None:
            inst = copy.copy(spec)
            inst.r_isco = float(r_isco)
            return inst
        return spec
    if isinstance(spec, str):
        if spec not in REGISTRY:
            raise ValueError(
                f"unknown colour_correction {spec!r}; "
                f"choices: {sorted(REGISTRY)}, or pass a class/value/callable.")
        cls = REGISTRY[spec]
        # If the class needs r_isco, pass it (and any default args).
        try:
            inst = cls(r_isco=r_isco) if 'r_isco' in inspect.signature(cls).parameters else cls()
        except TypeError:
            inst = cls()
        return inst
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return Constant(float(spec))
    if callable(spec):
        return _CallableWrap(spec)
    raise TypeError(
        f"colour_correction spec must be bool / str / number / callable / "
        f"ColourCorrection instance; got {type(spec).__name__}: {spec!r}")
