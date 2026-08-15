"""Numerical regression tests for GR_disc.log_likelihood.

These exist because the v2.0 release changes several defaults at once --
radial grid spacing, grid size, the Bessel approximation, and the sampling
parameterisation -- and nothing else in the package pins a number.

Two kinds of test here, and the distinction matters:

* **Property tests** assert an invariant that must hold in any environment
  (the reparameterisation cannot change the likelihood; the Bessel
  approximation must agree with scipy).  These are the valuable ones: they
  need no golden value and cannot drift.

* **Golden tests** pin an actual number.  They catch silent numerical drift,
  but a failure is not automatically a bug -- a different scipy could move the
  last few digits.  The tolerance is set well above that (1e-6 relative) so a
  failure means something real changed.

Every configuration is constructed with EXPLICIT keyword arguments, never by
relying on defaults, so that changing a default does not silently change what
these tests measure.

Run with:  pytest fitted/models/tests/test_likelihood_regression.py
"""

from pathlib import Path

import numpy as np
import pytest

from astropy import units

from fitted.data import Data_Set
from fitted.models import GR_disc


# --------------------------------------------------------------------------
# Fixture: a small AT2019dsg X-ray dataset built from the file that ships
# with the package, so the test is self-contained.
# --------------------------------------------------------------------------
_XRAY_FILE = Path(__file__).parents[2] / "examples" / "2019dsg_Xray.txt"
_D_MPC = 236.4  # AT2019dsg luminosity distance used by the shipped example


@pytest.fixture(scope="module")
def data():
    if not _XRAY_FILE.exists():                     # pragma: no cover
        pytest.skip(f"missing example data: {_XRAY_FILE}")
    arr = np.genfromtxt(_XRAY_FILE, delimiter="", skip_header=3)
    band = np.genfromtxt(_XRAY_FILE, delimiter="", skip_header=3,
                         usecols=(1), dtype=str)
    scale = 4 * np.pi * (_D_MPC * units.Mpc.to(units.cm)) ** 2
    t_x, l_x, e_x = arr[:, 0][band == "X"], arr[:, 2][band == "X"] * scale, \
        arr[:, 3][band == "X"] * scale
    t_ul, l_ul = arr[:, 0][band == "XUL"], arr[:, 2][band == "XUL"] * scale
    return Data_Set(
        args_X=[[t_x, l_x, e_x, [0.3, 10]]], bands_X=["Swift XRT"],
        args_X_upperlim=[[t_ul, l_ul, 3 * np.ones_like(l_ul), [0.3, 10]]],
        bands_X_upperlim=["Swift XRT UL"],
    )


# Parameters in linear space, and the same physical point expressed in the
# log / cosine sampling variables.
PARS_LINEAR = np.array([7.0, 0.01, 0.05, 30.0, 15.0, -2.0, 70.0,
                        43.3, 67.0, 5 / 3, 4.8])
PARS_LOG = PARS_LINEAR.copy()
PARS_LOG[2] = np.log10(PARS_LINEAR[2])            # m_disc  -> log_m_disc
PARS_LOG[3] = np.log10(PARS_LINEAR[3])            # r0      -> log_r0
PARS_LOG[4] = np.log10(PARS_LINEAR[4])            # tvi     -> log_tvi
PARS_LOG[6] = np.cos(np.deg2rad(PARS_LINEAR[6]))  # incl    -> cos_incl


def _model(data, **kw):
    """Build a model with EVERY relevant option pinned explicitly.

    The baseline here is the pre-2.0 released configuration.  Each test then
    overrides only what it is actually testing.  This matters: an earlier
    version of this helper left the fit_log_* flags unset, and when the
    package defaults flipped to True the "linear" baseline silently began
    interpreting linear parameters as log ones.  Nothing in the helper may
    depend on a package default.
    """
    base = dict(data=data, decay_type="pl", rise=False,
                radial_grid_spacing="linear", default_N=3000,
                use_iv_approximation=False,
                fit_log_m_disc=False, fit_log_r0=False,
                fit_log_tvi=False, fit_cos_incl=False)
    base.update(kw)
    return GR_disc(**base)


# ==========================================================================
# Property tests -- invariants, no golden values
# ==========================================================================
def test_reparameterisation_leaves_likelihood_unchanged(data):
    """log/cos sampling must be a pure change of variable.

    Sampling log10(m_disc) instead of m_disc changes the *prior*, but at a
    fixed physical point the likelihood is identical by construction.  If this
    fails, convert_parameters or the unpacking in log_likelihood is wrong, and
    every fit using the new defaults is quietly evaluating the wrong model.
    """
    linear = _model(data, radial_grid_spacing="linear", default_N=3000)
    logged = _model(data, radial_grid_spacing="linear", default_N=3000,
                    fit_log_m_disc=True, fit_log_r0=True,
                    fit_log_tvi=True, fit_cos_incl=True)
    assert linear.log_likelihood(PARS_LINEAR) == pytest.approx(
        logged.log_likelihood(PARS_LOG), abs=1e-8)


def test_iv_approximation_matches_scipy(data):
    """The Bessel approximation must not move the likelihood meaningfully."""
    off = _model(data, radial_grid_spacing="geometric", default_N=1000,
                 use_iv_approximation=False)
    on = _model(data, radial_grid_spacing="geometric", default_N=1000,
                use_iv_approximation=True, iv_approximation_accuracy="high")
    assert on.log_likelihood(PARS_LINEAR) == pytest.approx(
        off.log_likelihood(PARS_LINEAR), rel=1e-4)


def test_likelihood_is_finite_at_zero_spin(data):
    """a_bh = 0 must not raise or return nan.

    The vendored ray tracer has no Schwarzschild backend and numerical_disc_py
    floors |a|; GR_disc is analytic and should simply work.  Guards against a
    regression where a sampler visiting a_bh = 0 kills a chain.
    """
    m = _model(data, radial_grid_spacing="geometric", default_N=1000)
    p = PARS_LINEAR.copy()
    p[1] = 0.0
    assert np.isfinite(m.log_likelihood(p))


# ==========================================================================
# Golden tests -- pinned numbers, to catch silent drift
# ==========================================================================
# Generated with numpy 2.4.4 / scipy 1.17.1.  A change here means the
# numerics moved; investigate before updating the constant.
GOLDEN_RELEASED_CONFIG = -60.425378651425   # linear N=3000, linear parameters
GOLDEN_NEW_DEFAULTS = -60.892741718555      # geometric N=1000, log/cos parameters


def test_golden_released_configuration(data):
    """Pins the pre-2.0 numerics, so the legacy path cannot drift unnoticed."""
    m = _model(data, radial_grid_spacing="linear", default_N=3000,
               use_iv_approximation=False)
    assert m.log_likelihood(PARS_LINEAR) == pytest.approx(
        GOLDEN_RELEASED_CONFIG, rel=1e-6)


def test_golden_new_defaults(data):
    """Pins the v2.0 configuration.

    iv approximation is deliberately OFF here so the value is deterministic
    across environments with and without numba; the approximation is covered
    by test_iv_approximation_matches_scipy instead.
    """
    m = _model(data, radial_grid_spacing="geometric", default_N=1000,
               use_iv_approximation=False,
               fit_log_m_disc=True, fit_log_r0=True,
               fit_log_tvi=True, fit_cos_incl=True)
    assert m.log_likelihood(PARS_LOG) == pytest.approx(
        GOLDEN_NEW_DEFAULTS, rel=1e-6)
