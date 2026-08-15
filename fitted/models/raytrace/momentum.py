"""Photon 4-momentum in Boyer-Lindquist coordinates.

For a null geodesic in Kerr with E = 1, L_z = ξ, Q = η·E², the
contravariant BL components are (Carter separation):

    Σ p^t     = (r² + a²) P / Δ + a (ξ − a sin²θ)
    Σ p^r     = s_r · √R(r)
    Σ p^θ     = s_θ · √Θ(θ)
    Σ p^φ     = a P / Δ + ξ / sin²θ − a

with

    P(r)      = r² + a² − a ξ
    Δ(r)      = r² − 2 r + a²
    Σ(r, θ)   = r² + a² cos²θ
    R(r)      = P² − Δ · (η + (ξ − a)²)
    Θ(θ)      = η + a² cos²θ − ξ² cot²θ

Sign conventions:
    s_r = sign(dr/dλ) in forward Mino-time direction (±1)
    s_θ = sign(dθ/dλ) in forward Mino-time direction (±1)

The metric signature is (−, +, +, +); p^t is positive (forward-in-time).

See ./POLARIZATION_CONVENTIONS.md §3 for the full spec.
"""

import numpy as np


def photon_p_contravariant(a, xi, eta, r, theta, s_r, s_theta):
    """Photon contravariant 4-momentum in Boyer-Lindquist (t, r, θ, φ).

    Parameters
    ----------
    a        : float — Kerr spin.
    xi, eta  : scalar or array-like — conserved quantities L_z/E, Q/E².
    r, theta : scalar or array-like — BL event coordinates.
    s_r      : ±1 scalar or array-like — sign of dr/dλ.
    s_theta  : ±1 scalar or array-like — sign of dθ/dλ.

    All array-like inputs are broadcast together to a common shape `S`.

    Returns
    -------
    p : ndarray, shape (S, 4)
        Contravariant components (p^t, p^r, p^θ, p^φ) at the given event,
        with the last axis indexing the coordinate direction.

    Notes
    -----
    R(r) or Θ(θ) being (slightly) negative from floating-point noise at
    turning points is clipped to zero; the corresponding p^r / p^θ
    evaluates to zero at the turning point, which is the correct
    analytic value.

    At the horizon r = r_+, Δ → 0 and p^t, p^φ formally diverge — this
    is the well-known BL coordinate pathology.  The caller is
    responsible for not asking for p^μ on the horizon.
    """
    a = float(a)
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    s_r = np.asarray(s_r, dtype=np.float64)
    s_theta = np.asarray(s_theta, dtype=np.float64)

    # Broadcast to common shape
    xi, eta, r, theta, s_r, s_theta = np.broadcast_arrays(
        xi, eta, r, theta, s_r, s_theta
    )

    sinth = np.sin(theta)
    costh = np.cos(theta)
    sin2 = sinth * sinth

    Sigma = r * r + (a * costh) ** 2
    Delta = r * r - 2.0 * r + a * a
    P = r * r + a * a - a * xi

    # Carter potentials.  Clip to zero at turning points.
    R_pot = P * P - Delta * (eta + (xi - a) ** 2)
    R_pot = np.where(R_pot > 0.0, R_pot, 0.0)
    # Θ(θ) = η + a² cos²θ − ξ² cot²θ.  At the poles sin θ = 0 and
    # cot²θ = ∞; valid null geodesics have ξ² cos²θ / sin²θ ≤ η + a² cos²θ
    # so the product stays finite, but numerically we avoid the raw 1/sin²
    # by regrouping as (η sin²θ + a² cos²θ sin²θ − ξ² cos²θ) / sin²θ.
    # For any point strictly away from the poles this matches the naive
    # expression.
    sin2_safe = np.where(sin2 > 0.0, sin2, np.nan)
    Theta_pot = (eta * sin2 + (a * costh) ** 2 * sin2 - (xi * costh) ** 2) / sin2_safe
    Theta_pot = np.where(Theta_pot > 0.0, Theta_pot, 0.0)

    sqrtR = np.sqrt(R_pot)
    sqrtTh = np.sqrt(Theta_pot)

    # Σ p^μ (Carter form)
    Sigma_pt = (r * r + a * a) * P / Delta + a * (xi - a * sin2)
    Sigma_pr = s_r * sqrtR
    Sigma_pth = s_theta * sqrtTh
    Sigma_pphi = a * P / Delta + xi / sin2_safe - a

    p_t = Sigma_pt / Sigma
    p_r = Sigma_pr / Sigma
    p_theta = Sigma_pth / Sigma
    p_phi = Sigma_pphi / Sigma

    return np.stack([p_t, p_r, p_theta, p_phi], axis=-1)


def photon_p_covariant(a, xi, eta, r, theta, s_r, s_theta):
    """Photon covariant 4-momentum p_μ in Boyer-Lindquist.

    With E = 1, the conserved components are
        p_t   = −E = −1
        p_φ   = +L_z = +ξ
    The remaining components come from the Carter split
        p_r   = s_r · √R(r) / Δ(r)
        p_θ   = s_θ · √Θ(θ)

    Same signature and broadcasting rules as `photon_p_contravariant`.
    """
    a = float(a)
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    s_r = np.asarray(s_r, dtype=np.float64)
    s_theta = np.asarray(s_theta, dtype=np.float64)

    xi, eta, r, theta, s_r, s_theta = np.broadcast_arrays(
        xi, eta, r, theta, s_r, s_theta
    )

    sinth = np.sin(theta)
    costh = np.cos(theta)
    sin2 = sinth * sinth

    Delta = r * r - 2.0 * r + a * a
    P = r * r + a * a - a * xi

    R_pot = P * P - Delta * (eta + (xi - a) ** 2)
    R_pot = np.where(R_pot > 0.0, R_pot, 0.0)
    sin2_safe = np.where(sin2 > 0.0, sin2, np.nan)
    Theta_pot = (eta * sin2 + (a * costh) ** 2 * sin2
                  - (xi * costh) ** 2) / sin2_safe
    Theta_pot = np.where(Theta_pot > 0.0, Theta_pot, 0.0)

    p_t = -np.ones_like(r)
    p_r = s_r * np.sqrt(R_pot) / Delta
    p_theta = s_theta * np.sqrt(Theta_pot)
    p_phi = xi * np.ones_like(r)

    return np.stack([p_t, p_r, p_theta, p_phi], axis=-1)


def bl_metric(a, r, theta):
    """Boyer-Lindquist metric g_{μν} and inverse g^{μν} at (r, θ).

    Signature (−, +, +, +).  Returns two (4, 4) arrays broadcast over
    the input shape.  For scalar (r, θ) the return shape is (4, 4);
    for array (r, θ) of shape `S` the return shapes are (S, 4, 4).
    """
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    r, theta = np.broadcast_arrays(r, theta)

    sinth = np.sin(theta)
    costh = np.cos(theta)
    sin2 = sinth * sinth

    Sigma = r * r + (a * costh) ** 2
    Delta = r * r - 2.0 * r + a * a
    A = (r * r + a * a) ** 2 - a * a * sin2 * Delta

    shape = r.shape
    g = np.zeros(shape + (4, 4), dtype=np.float64)
    g[..., 0, 0] = -(1.0 - 2.0 * r / Sigma)
    g[..., 0, 3] = -2.0 * a * r * sin2 / Sigma
    g[..., 3, 0] = g[..., 0, 3]
    g[..., 1, 1] = Sigma / Delta
    g[..., 2, 2] = Sigma
    g[..., 3, 3] = A * sin2 / Sigma

    sin2_safe = np.where(sin2 > 0.0, sin2, np.nan)
    g_inv = np.zeros_like(g)
    g_inv[..., 0, 0] = -A / (Sigma * Delta)
    g_inv[..., 0, 3] = -2.0 * a * r / (Sigma * Delta)
    g_inv[..., 3, 0] = g_inv[..., 0, 3]
    g_inv[..., 1, 1] = Delta / Sigma
    g_inv[..., 2, 2] = 1.0 / Sigma
    g_inv[..., 3, 3] = (Delta - a * a * sin2) / (Sigma * Delta * sin2_safe)
    return g, g_inv


def null_check(a, r, theta, p_contra):
    """Return g_{μν} p^μ p^ν for each event — should be ≈ 0.

    p_contra : (..., 4) array of contravariant components.
    Returns an array of shape (...).
    """
    g, _ = bl_metric(a, r, theta)
    return np.einsum('...ij,...i,...j->...', g, p_contra, p_contra)
