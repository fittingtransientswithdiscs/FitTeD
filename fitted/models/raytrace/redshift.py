"""Photon redshift factor and static-observer kinematics in Kerr.

Single source of truth for

    g = ω_obs / ω_em,    ω_X = −(u_X · p_X)_g_X

so that intensity-only and polarized pipelines agree by construction. Both
``polarized_trace`` and the lean intensity orchestrator import from here.

Notes
-----
``static_observer_4velocity`` is the asymptotic-static observer 4-velocity
``u^μ = (1/√(−g_tt), 0, 0, 0)``.  For r_obs ≫ M this is within O(M/r_obs)
of (1, 0, 0, 0).  At finite r_obs inside the ergoregion (only possible for
a > 0 and θ near the equator) g_tt > 0 and a static observer does not exist
— callers should switch to a ZAMO/LNRF tetrad in that regime.
"""

import numpy as np

from .momentum import bl_metric


def _dot(g, a, b):
    """Lorentzian inner product g_{μν} a^μ b^ν, broadcast over leading axes."""
    return np.einsum('...ij,...i,...j->...', g, a, b)


def static_observer_4velocity(a, r_obs, theta_obs):
    """4-velocity of an asymptotic static observer at (r_obs, θ_obs).

    Parameters
    ----------
    a : float
        BH spin in units of M.
    r_obs, theta_obs : array_like
        Observer Boyer–Lindquist coordinates (broadcast together).

    Returns
    -------
    u : ndarray, shape (..., 4)
        Static-observer 4-velocity in BL components.
    """
    g, _ = bl_metric(a, r_obs, theta_obs)
    shape = np.asarray(r_obs).shape
    u = np.zeros(shape + (4,), dtype=np.float64)
    u[..., 0] = 1.0 / np.sqrt(-g[..., 0, 0])
    return u


def redshift_factor(a, u_em, p_em, r_em, theta_em,
                       u_obs, p_obs, r_obs, theta_obs):
    """Photon redshift factor g = ω_obs / ω_em.

    Computes ω_em = −(u_em · p_em) at (r_em, θ_em) and ω_obs = −(u_obs · p_obs)
    at (r_obs, θ_obs) using the BL metric at each event, then returns the
    ratio.  Invalid / non-finite inputs propagate as NaN without warnings.

    Parameters
    ----------
    a : float
    u_em, p_em : (..., 4)
        Emitter 4-velocity and photon 4-momentum at emission.
    r_em, theta_em : (...)
        Emission BL coordinates (broadcast with the leading axes of u_em, p_em).
    u_obs, p_obs : (..., 4)
        Observer 4-velocity and photon 4-momentum at the observer.
    r_obs, theta_obs : (...)
        Observer BL coordinates.

    Returns
    -------
    g : ndarray
        ω_obs / ω_em, broadcast to the common leading shape.

    Notes
    -----
    For an asymptotic static observer with u_obs ≈ (1, 0, 0, 0) we have
    ω_obs = p^t (BL) = −E_∞ in geometrized units, so g reduces to the usual
    "redshift through E_∞ / E_em" definition.  See Bardeen, Press & Teukolsky
    (1972) §II for the conventions.
    """
    g_em, _  = bl_metric(a, r_em,  theta_em)
    g_obs, _ = bl_metric(a, r_obs, theta_obs)
    omega_em  = -_dot(g_em,  u_em,  p_em)
    omega_obs = -_dot(g_obs, u_obs, p_obs)
    with np.errstate(invalid='ignore', divide='ignore'):
        return omega_obs / omega_em
