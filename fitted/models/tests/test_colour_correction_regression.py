"""
Pre-refactor regression baseline for the colour-correction overhaul.

Captures default-colour-correction outputs (SED, UV light curve, X-ray
light curve) for both gr_disc (analytic) and gr_disc_plus (kerrgeo)
backends at fixed parameters.  After the refactor, the same fixtures
must reproduce the same numbers to ~1e-12.

Two modes:
    --capture          save the baseline arrays to disk (run before refactor)
    (no flag)          load and compare (run after refactor)

The baseline file lives next to this script and is intended to be a
disposable reference, not committed long-term -- once the refactor lands
and tests pass the file can be removed.

Usage:
    python test_colour_correction_regression.py --capture       # before
    python test_colour_correction_regression.py                 # after
"""

from __future__ import annotations

import argparse
import io
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "_colour_correction_baseline.npz"


def _build_fixtures():
    """Build a deterministic (parameters, models, frequencies, times) set."""
    sys.path.insert(0, str(HERE.parents[2]))   # parent of fitted/ pkg
    import fitted

    # Common parameters: a moderate-spin TDE-style configuration.
    pars = dict(
        log_mh=6.5, a_bh=0.5, m_disc=0.05, r0=30.0, tvi=15.0, t0=-2.0, incl=60.0,
    )
    # Frequencies: UV + X-ray.
    keV_to_Hz = fitted.constants.keV_to_Hz
    vs = np.array([
        2.4e15, 4.5e15, 6e14,                             # near-UV
        0.3 * keV_to_Hz, 1.0 * keV_to_Hz, 5.0 * keV_to_Hz, # X-ray
    ])
    times = np.linspace(20, 500, 8)

    # Models with default colour_correction=True
    m_simple = fitted.models.GR_disc()
    m_kerrgeo = fitted.models.GR_disc_plus()

    return pars, vs, times, m_simple, m_kerrgeo


def _compute_outputs(pars, vs, times, m_simple, m_kerrgeo):
    """Run the standard prediction methods on both models."""
    import fitted
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # gr_disc
        out['gr_disc.model_SED']  = np.asarray(m_simple.model_SED(times[3], vs=vs, **pars))
        out['gr_disc.model_SEDs'] = np.asarray(m_simple.model_SEDs(times, vs=vs, **pars))
        # X-ray light curve uses energy band; UV uses a single frequency.
        out['gr_disc.model_X_03_10keV'] = np.asarray(
            m_simple.model_X(times, El=0.3, Eh=10.0, **pars))
        out['gr_disc.model_UV_2.4e15'] = np.asarray(
            m_simple.model_UV(times, v=2.4e15, **pars))

        # gr_disc_plus (kerrgeo)
        out['gr_disc_plus.model_SEDs'] = np.asarray(
            m_kerrgeo.model_SEDs(times, vs=vs, **pars))
        out['gr_disc_plus.model_X_03_10keV'] = np.asarray(
            m_kerrgeo.model_X(times, El=0.3, Eh=10.0, **pars))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true",
                     help="save baseline (run before refactor)")
    ap.add_argument("--rtol", type=float, default=1e-12,
                     help="tolerance for after-refactor comparison")
    args = ap.parse_args()

    print("Building fixtures...")
    pars, vs, times, m_simple, m_kerrgeo = _build_fixtures()

    # Suppress fitted's banner.
    with redirect_stdout(io.StringIO()):
        out = _compute_outputs(pars, vs, times, m_simple, m_kerrgeo)

    if args.capture:
        np.savez(BASELINE, **out)
        print(f"Captured baseline -> {BASELINE}")
        print(f"Quantities: {sorted(out)}")
        for k, v in out.items():
            print(f"  {k:<40s} shape={str(v.shape):<15s} "
                  f"min={np.nanmin(v):.3e} max={np.nanmax(v):.3e}")
        return

    if not BASELINE.exists():
        print(f"No baseline at {BASELINE}; run with --capture first.")
        sys.exit(2)

    ref = np.load(BASELINE)
    n_pass = n_fail = 0
    print(f"\nComparing against baseline at rtol={args.rtol:.0e}:")
    for k in sorted(out):
        a = np.asarray(out[k])
        b = np.asarray(ref[k])
        if a.shape != b.shape:
            print(f"  [FAIL] {k:<40s} shape {a.shape} vs {b.shape}")
            n_fail += 1
            continue
        denom = np.maximum(np.abs(a), np.abs(b))
        denom = np.where(denom == 0, 1.0, denom)
        rel = np.abs(a - b) / denom
        rel = np.where(np.isfinite(rel), rel, 0)
        max_rel = float(np.nanmax(rel))
        ok = max_rel <= args.rtol
        flag = "OK  " if ok else "FAIL"
        print(f"  [{flag}] {k:<40s}  max_rel = {max_rel:.3e}")
        if ok:
            n_pass += 1
        else:
            n_fail += 1
    print(f"\n  {n_pass} pass / {n_fail} fail")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
