"""Bardeen camera transform.

Observer at r=∞, inclination i ∈ (0, π) from the spin axis.  Pixel
coords (α, β) in the observer's image plane:
    α  — horizontal (perpendicular to the projected spin axis)
    β  — vertical   (parallel   to the projected spin axis)

Conserved quantities per pixel:
    ξ  =  L_z / E   =  -α sin i
    η  =  Q   / E²  =   β² + (α² - a²) cos² i

Observer polar coord:
    μ_obs = cos i

Sign convention for the forward photon's dμ/dλ at the observer
(needed by the polar Mino-time step):
    s_μ_forward_obs = +sign(β)  under the convention adopted here.

This convention is validated against a DOP853 brute-force ray-tracer
in `camera_brute_force.py`; if the sign is flipped relative to the
brute-force truth the tests will surface a polar-Mino mismatch, and
the constant below can be flipped in one place.

All functions are pure-numpy and fully vectorised.
"""

import numpy as np

# Sign of s_μ_forward_at_observer in terms of sign(β).
#
#   p_θ_forward_at_observer = +β       (Bardeen, observer at r=∞)
#   dθ/dλ_Mino_forward      = p_θ      (Carter separability)
#   dμ/dλ_forward           = -sin θ · dθ/dλ = -sin i · β
#
# Hence  s_μ_forward_obs  =  -sign(β).
S_MU_SIGN = -1.0


def bardeen_transform(a, i, alpha, beta):
    """Map image-plane pixels to Kerr conserved quantities.

    Parameters
    ----------
    a      : float
        Kerr spin.
    i      : float
        Observer inclination (radians, ∈ (0, π)).
    alpha  : array_like
        Image-plane horizontal coordinates.
    beta   : array_like
        Image-plane vertical coordinates.

    Returns
    -------
    xi, eta, mu_obs, s_mu_obs : ndarrays of common broadcast shape
        Conserved quantities and the polar-velocity sign at the
        observer used downstream by the polar Mino-time step.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    alpha, beta = np.broadcast_arrays(alpha, beta)

    sin_i = np.sin(i)
    cos_i = np.cos(i)
    a = float(a)

    xi = -alpha * sin_i
    eta = beta * beta + (alpha * alpha - a * a) * cos_i * cos_i

    mu_obs = np.full_like(alpha, cos_i, dtype=np.float64)
    # +1 for β ≥ 0, −1 for β < 0 (modulo S_MU_SIGN global).
    s_mu_obs = np.where(beta >= 0.0, S_MU_SIGN, -S_MU_SIGN)
    return xi, eta, mu_obs, s_mu_obs
