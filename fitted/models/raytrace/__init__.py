"""Vendored minimal Kerr ray tracer for FitTeD.

Extracted subset of kerrgeo -- only the modules required by the disc SED
path (the vectorised ``make_batch`` route through ``intensity_trace``).
See PROVENANCE.md for the upstream commit and what was removed.
"""
from .intensity_trace import intensity_trace
from .thermal import planck_weight
from .observers.asymptotic import AsymptoticObserver
from .targets.equatorial import EquatorialDisk
from .sources.keplerian import KeplerianDisk
from .sources.kerr_disk import KerrDisk

__all__ = ["intensity_trace", "planck_weight", "AsymptoticObserver",
           "EquatorialDisk", "KeplerianDisk", "KerrDisk"]
