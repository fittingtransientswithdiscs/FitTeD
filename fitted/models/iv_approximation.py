"""
Custom approximation for scipy.special.iv() (modified Bessel function)

This module provides fast approximations for I_v(x) optimized for v = 1/3.
Uses a hybrid approach with different methods for different argument ranges:
- Small x (0 ≤ x < 0.1): Power series expansion
- Medium x (0.1 ≤ x < 10): Chebyshev polynomial approximation
- Large x (x ≥ 10): Asymptotic expansion
"""

import os
import numpy as np
from scipy.special import gamma, iv
from pathlib import Path
import warnings

# Numba's on-disk cache must be usable BEFORE numba is imported.
#
# With cache=False every process recompiles all four kernels from scratch --
# measured at ~2.7 s of LLVM compilation each.  In a multiprocessing pool that
# is paid once per worker, which made use_iv_approximation=True *slower* in
# parallel than in serial on a 100-step AT2019dsg chain.
#
# cache=True writes .nbi/.nbc into __pycache__ beside this file.  On a
# read-only install that fails, so point numba at a user cache directory
# instead of silently losing caching.
if not os.environ.get("NUMBA_CACHE_DIR"):
    if not os.access(Path(__file__).parent, os.W_OK):
        os.environ["NUMBA_CACHE_DIR"] = os.path.join(
            os.path.expanduser("~"), ".cache", "fitted_numba")

# OPTIMIZATION: Try to import Numba for JIT compilation (optional)
try:
    from numba import jit, types
    from numba import float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a dummy decorator that does nothing if Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Fixed order used in FitTeD
V_ORDER = 1/3  # order = 1/(4*alpha) where alpha = 3/4

# Pre-compute gamma values for power series
def _precompute_gamma_values(max_k=30):
    """
    Pre-compute gamma function values for power series
    
    Returns array of Γ(v + k + 1) for k = 0, 1, ..., max_k
    """
    gamma_values = np.zeros(max_k + 1)
    for k in range(max_k + 1):
        gamma_values[k] = gamma(V_ORDER + k + 1)
    return gamma_values

# Pre-compute once at module level
_GAMMA_VALUES = _precompute_gamma_values(max_k=30)

# Pre-compute factorial values for power series
_FACTORIAL = np.ones(31)
for k in range(1, 31):
    _FACTORIAL[k] = _FACTORIAL[k-1] * k


# OPTIMIZATION: JIT-compiled core computation for power series
@jit(nopython=True, cache=True)
def _iv_power_series_core(x_nonzero, v, gamma_values, factorial_values, max_terms, tol):
    """
    JIT-compiled core computation for power series.
    This is the hot loop that benefits most from JIT compilation.
    """
    n = len(x_nonzero)
    result = np.zeros(n)
    
    # Pre-factor: (x/2)^v
    x_half = x_nonzero / 2.0
    prefactor = np.power(x_half, v)
    
    # Initialize series sum
    series_sum = np.zeros(n)
    
    # Compute series terms
    x_half_sq = x_nonzero * x_nonzero / 4.0  # (x/2)^2
    
    # Use pre-computed gamma and factorial values
    max_k = min(max_terms, len(gamma_values) - 1, len(factorial_values) - 1)
    
    for k in range(max_k + 1):
        # Term: ((x/2)^(2k)) / (k! * Γ(v+k+1))
        x_power = np.power(x_half_sq, k)
        term = x_power / (factorial_values[k] * gamma_values[k])
        series_sum += term
        
        # Early termination check (if terms become negligible)
        if k > 5:  # Need at least a few terms
            max_term = 0.0
            max_sum = 0.0
            for i in range(n):
                abs_term = abs(term[i])
                abs_sum = abs(series_sum[i])
                if abs_term > max_term:
                    max_term = abs_term
                if abs_sum > max_sum:
                    max_sum = abs_sum
            if max_sum > 0 and max_term < tol * max_sum:
                break
    
    # Combine prefactor and series
    result = prefactor * series_sum
    
    return result


def iv_power_series(v, x, max_terms=20, tol=1e-10):
    """
    Power series expansion for I_v(x) for small x
    
    I_v(x) ≈ (x/2)^v * Σ_{k=0}^max_terms [((x/2)^(2k)) / (k! * Γ(v+k+1))]
    
    Parameters:
    -----------
    v : float
        Order (should be 1/3)
    x : array-like
        Arguments (must be >= 0, typically < 0.1)
    max_terms : int
        Maximum number of terms in series
    tol : float
        Tolerance for early termination
    
    Returns:
    --------
    result : array
        Approximate I_v(x) values
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Handle edge case: x = 0
    result = np.zeros_like(x)
    zero_mask = (x == 0)
    if np.all(zero_mask):
        return result
    
    # Filter out zeros
    x_nonzero = x[~zero_mask]
    
    # OPTIMIZATION: Use JIT-compiled core if Numba is available
    if NUMBA_AVAILABLE and len(x_nonzero) > 0:
        result[~zero_mask] = _iv_power_series_core(
            x_nonzero, v, _GAMMA_VALUES, _FACTORIAL, max_terms, tol
        )
    else:
        # Fallback to original NumPy implementation
        # Pre-factor: (x/2)^v
        x_half = x_nonzero / 2.0
        prefactor = np.power(x_half, v)
        
        # Initialize series sum
        series_sum = np.zeros_like(x_nonzero)
        
        # Compute series terms
        x_half_sq = x_nonzero * x_nonzero / 4.0  # (x/2)^2
        
        # Use pre-computed gamma and factorial values
        max_k = min(max_terms, len(_GAMMA_VALUES) - 1, len(_FACTORIAL) - 1)
        
        for k in range(max_k + 1):
            # Term: ((x/2)^(2k)) / (k! * Γ(v+k+1))
            term = np.power(x_half_sq, k) / (_FACTORIAL[k] * _GAMMA_VALUES[k])
            series_sum += term
            
            # Early termination check (if terms become negligible)
            if k > 5:  # Need at least a few terms
                max_term = np.max(np.abs(term))
                max_sum = np.max(np.abs(series_sum))
                if max_sum > 0 and max_term < tol * max_sum:
                    break
        
        # Combine prefactor and series
        result[~zero_mask] = prefactor * series_sum
    
    return result


# Chebyshev coefficients (will be loaded from file if available)
_CHEBYSHEV_COEFFS = None
_CHEBYSHEV_RANGE = (0.1, 10.0)
_CHEBYSHEV_N_TERMS = 35

def _load_chebyshev_coefficients():
    """Load pre-computed Chebyshev coefficients if available"""
    global _CHEBYSHEV_COEFFS
    if _CHEBYSHEV_COEFFS is not None:
        return
    
    # Try to load from current directory (where this module is located)
    coeff_file = Path(__file__).parent / 'chebyshev_coefficients.npy'
    if coeff_file.exists():
        try:
            _CHEBYSHEV_COEFFS = np.load(coeff_file)
            _CHEBYSHEV_N_TERMS = len(_CHEBYSHEV_COEFFS)
            return
        except Exception as e:
            warnings.warn(f"Failed to load Chebyshev coefficients: {e}")
    
    # If not available, will compute on the fly (slower)
    _CHEBYSHEV_COEFFS = None


# OPTIMIZATION: JIT-compiled core computation for Chebyshev
@jit(nopython=True, cache=True)
def _iv_chebyshev_core(x_transformed, coeffs):
    """
    JIT-compiled Clenshaw's recurrence for Chebyshev polynomial evaluation.
    This is the hot loop that benefits most from JIT compilation.
    """
    n = len(coeffs)
    n_x = len(x_transformed)
    result = np.zeros(n_x)
    
    # Initialize arrays for all x values simultaneously
    b_kp2 = np.zeros(n_x)  # b_{n+1} = 0 for all x
    b_kp1 = np.full(n_x, coeffs[n-1])  # b_n = c_n for all x
    
    # Recurrence: b_k = c_k + 2*x*b_{k+1} - b_{k+2} for k = n-1, ..., 1
    # Process all x values simultaneously
    for k in range(n - 1, 0, -1):
        b_k = coeffs[k] + 2 * x_transformed * b_kp1 - b_kp2
        b_kp2 = b_kp1
        b_kp1 = b_k
    
    # Final result: c_0 + x*b_1 - b_2 (for all x simultaneously)
    result = coeffs[0] + x_transformed * b_kp1 - b_kp2
    
    return result


def iv_chebyshev(v, x, range_min=0.1, range_max=10.0, n_terms=None):
    """
    Chebyshev polynomial approximation for I_v(x) in [range_min, range_max]
    
    Uses Clenshaw's recurrence for stable evaluation
    
    Parameters:
    -----------
    v : float
        Order (should be 1/3)
    x : array-like
        Arguments (should be in [range_min, range_max])
    range_min, range_max : float
        Approximation range
    n_terms : int, optional
        Number of Chebyshev terms (uses pre-computed if available)
    
    Returns:
    --------
    result : array
        Approximate I_v(x) values
    """
    from scipy.special import iv
    
    x = np.asarray(x, dtype=np.float64)
    
    # Load coefficients if not already loaded
    _load_chebyshev_coefficients()
    
    # If coefficients not available, fall back to scipy
    if _CHEBYSHEV_COEFFS is None:
        warnings.warn("Chebyshev coefficients not available, using scipy")
        return iv(v, x)
    
    # Use pre-computed coefficients
    coeffs = _CHEBYSHEV_COEFFS
    if n_terms is not None:
        coeffs = coeffs[:n_terms]
    n = len(coeffs)
    
    # Transform x from [range_min, range_max] to [-1, 1]
    x_transformed = (2 * x - (range_max + range_min)) / (range_max - range_min)
    
    # OPTIMIZATION: Use JIT-compiled core if Numba is available
    if NUMBA_AVAILABLE:
        result = _iv_chebyshev_core(x_transformed, coeffs)
    else:
        # Fallback to original NumPy implementation
        # Vectorized Clenshaw's recurrence for all x values at once
        # Standard Clenshaw algorithm for Chebyshev polynomials:
        # b_{n+1} = 0, b_n = c_n
        # b_k = c_k + 2*x*b_{k+1} - b_{k+2} for k = n-1, ..., 1
        # Result = c_0 + x*b_1 - b_2
        
        # Initialize arrays for all x values simultaneously
        b_kp2 = np.zeros_like(x_transformed)  # b_{n+1} = 0 for all x
        b_kp1 = np.full_like(x_transformed, coeffs[n-1])  # b_n = c_n for all x
        
        # Recurrence: b_k = c_k + 2*x*b_{k+1} - b_{k+2} for k = n-1, ..., 1
        # Process all x values simultaneously
        for k in range(n - 1, 0, -1):
            b_k = coeffs[k] + 2 * x_transformed * b_kp1 - b_kp2
            b_kp2 = b_kp1
            b_kp1 = b_k
        
        # Final result: c_0 + x*b_1 - b_2 (for all x simultaneously)
        result = coeffs[0] + x_transformed * b_kp1 - b_kp2
    
    return result


# OPTIMIZATION: JIT-compiled core computation for asymptotic expansion
@jit(nopython=True, cache=True)
def _iv_asymptotic_moderate_core(x_moderate, v, n_terms):
    """
    JIT-compiled asymptotic expansion for moderate x (x <= 100).
    This is the hot loop that benefits most from JIT compilation.
    """
    n = len(x_moderate)
    result = np.zeros(n)
    
    # Pre-factor: e^x / sqrt(2πx)
    prefactor = np.exp(x_moderate) / np.sqrt(2 * np.pi * x_moderate)
    
    # Series terms
    series = np.ones(n)
    
    # Pre-compute (4v²-1) and related terms for efficiency
    v_sq_term = 4 * v * v - 1
    v_sq_9 = 4 * v * v - 9
    v_sq_25 = 4 * v * v - 25
    v_sq_49 = 4 * v * v - 49
    x_inv = 1.0 / x_moderate
    
    # Pre-compute powers of x_inv (vectorized - numba supports this)
    x_inv_sq = x_inv * x_inv
    x_inv_cubed = x_inv_sq * x_inv
    x_inv_4th = x_inv_cubed * x_inv
    
    # Compute series terms - optimized with pre-computed values
    term = np.ones(n)
    for k in range(1, n_terms):
        if k == 1:
            term = -v_sq_term * x_inv / 8.0
        elif k == 2:
            term = v_sq_term * v_sq_9 * x_inv_sq / 128.0
        elif k == 3:
            term = -v_sq_term * v_sq_9 * v_sq_25 * x_inv_cubed / 3072.0
        elif k == 4:
            term = v_sq_term * v_sq_9 * v_sq_25 * v_sq_49 * x_inv_4th / 98304.0
        else:
            # Higher terms: use recurrence relation (vectorized)
            coef = (4*v*v - (2*k-1)*(2*k-1)) / (8.0 * k)
            term = term * coef * x_inv
        
        series += term
        
        # Early termination if terms become negligible - vectorized check
        if k > 2:
            # Vectorized: check if all terms are negligible relative to series
            abs_term = np.abs(term)
            abs_series = np.abs(series)
            max_term = np.max(abs_term)
            max_series = np.max(abs_series)
            if max_series > 0 and max_term < 1e-12 * max_series:
                break
    
    result = prefactor * series
    return result


@jit(nopython=True, cache=True)
def _iv_asymptotic_large_core(x_large, v, n_terms):
    """
    JIT-compiled asymptotic expansion for large x (x > 100) in log space.
    This is the hot loop that benefits most from JIT compilation.
    """
    n = len(x_large)
    result = np.zeros(n)
    
    # Compute in log space: log(I_v(x)) = x - 0.5*log(2πx) + log(series)
    log_prefactor = x_large - 0.5 * np.log(2 * np.pi * x_large)
    
    # Series in log space (first term is 0 = log(1))
    log_series = np.zeros(n)
    v_sq_term = 4 * v * v - 1
    x_inv = 1.0 / x_large
    
    # First correction term: log(1 - (4v²-1)/(8x)) ≈ -(4v²-1)/(8x) for large x
    first_correction = -v_sq_term * x_inv / 8.0
    log_series = first_correction  # log(1 + ε) ≈ ε for small ε
    
    # For very large x, first term dominates, so we can stop here
    # But add more terms for better accuracy (vectorized)
    if n_terms > 1:
        second_correction = v_sq_term * (4*v*v - 9) * x_inv * x_inv / 128.0
        log_series += second_correction
    if n_terms > 2:
        # Third term for even better accuracy (only for very large x where it matters)
        third_correction = -v_sq_term * (4*v*v - 9) * (4*v*v - 25) * x_inv * x_inv * x_inv / 3072.0
        log_series += third_correction
    
    result = np.exp(log_prefactor + log_series)
    return result


def iv_asymptotic(v, x, n_terms=5):
    """
    Asymptotic expansion for I_v(x) for large x
    
    I_v(x) ≈ (e^x / sqrt(2πx)) * [1 - (4v²-1)/(8x) + ...]
    
    Parameters:
    -----------
    v : float
        Order (should be 1/3)
    x : array-like
        Arguments (should be >= 10 for good accuracy)
    n_terms : int
        Number of terms in asymptotic series
    
    Returns:
    --------
    result : array
        Approximate I_v(x) values
    """
    from scipy.special import iv
    
    x = np.asarray(x, dtype=np.float64)
    
    # Handle very small x (shouldn't happen, but be safe)
    mask = x > 1e-10
    result = np.zeros_like(x)
    
    if not np.any(mask):
        return result
    
    x_valid = x[mask]
    
    # For very large x, use log-space computation to avoid overflow
    large_x_mask_full = mask & (x > 100)  # Full mask for large x
    moderate_x_mask_full = mask & (x <= 100) & (x > 1e-10)  # Full mask for moderate x
    
    # Moderate x: regular computation
    if np.any(moderate_x_mask_full):
        x_moderate = x[moderate_x_mask_full]
        
        # OPTIMIZATION: Use JIT-compiled core if Numba is available
        if NUMBA_AVAILABLE:
            result[moderate_x_mask_full] = _iv_asymptotic_moderate_core(x_moderate, v, n_terms)
        else:
            # Fallback to original NumPy implementation
            # Pre-factor: e^x / sqrt(2πx)
            prefactor = np.exp(x_moderate) / np.sqrt(2 * np.pi * x_moderate)
            
            # Series terms
            series = np.ones_like(x_moderate)
            
            # Pre-compute (4v²-1) for efficiency
            v_sq_term = 4 * v * v - 1
            x_inv = 1.0 / x_moderate
            x_inv_sq = x_inv * x_inv  # Pre-compute x_inv^2
            x_inv_cubed = x_inv_sq * x_inv  # Pre-compute x_inv^3
            x_inv_4th = x_inv_cubed * x_inv  # Pre-compute x_inv^4
            
            # Pre-compute common factors
            v_sq_9 = 4*v*v - 9
            v_sq_25 = 4*v*v - 25
            v_sq_49 = 4*v*v - 49
            
            # Compute series terms - optimized with pre-computed values
            term = np.ones_like(x_moderate)  # Fix: was scalar 1.0, now array
            for k in range(1, n_terms):
                if k == 1:
                    term = -v_sq_term * x_inv / 8.0
                elif k == 2:
                    term = v_sq_term * v_sq_9 * x_inv_sq / 128.0
                elif k == 3:
                    term = -v_sq_term * v_sq_9 * v_sq_25 * x_inv_cubed / 3072.0
                elif k == 4:
                    term = v_sq_term * v_sq_9 * v_sq_25 * v_sq_49 * x_inv_4th / 98304.0
                else:
                    # Higher terms: use recurrence relation (vectorized)
                    coef = (4*v*v - (2*k-1)**2) / (8.0 * k)
                    term = term * coef * x_inv
                
                series += term
                
                # Early termination if terms become negligible (vectorized check)
                if k > 2:
                    max_term = np.max(np.abs(term))
                    max_series = np.max(np.abs(series))
                    if max_series > 0 and max_term < 1e-12 * max_series:
                        break
            
            result[moderate_x_mask_full] = prefactor * series
    
    # Large x: log-space computation
    if np.any(large_x_mask_full):
        x_large = x[large_x_mask_full]
        
        # OPTIMIZATION: Use JIT-compiled core if Numba is available
        if NUMBA_AVAILABLE:
            result[large_x_mask_full] = _iv_asymptotic_large_core(x_large, v, n_terms)
        else:
            # Fallback to original NumPy implementation
            # Compute in log space: log(I_v(x)) = x - 0.5*log(2πx) + log(series)
            log_prefactor = x_large - 0.5 * np.log(2 * np.pi * x_large)
            
            # Series in log space (first term is 0 = log(1))
            log_series = np.zeros_like(x_large)
            v_sq_term = 4 * v * v - 1
            x_inv = 1.0 / x_large
            
            # First correction term: log(1 - (4v²-1)/(8x)) ≈ -(4v²-1)/(8x) for large x
            first_correction = -v_sq_term * x_inv / 8.0
            log_series = first_correction  # log(1 + ε) ≈ ε for small ε
            
            # For very large x, first term dominates, so we can stop here
            # But add more terms for better accuracy
            if n_terms > 1:
                second_correction = v_sq_term * (4*v*v - 9) * x_inv * x_inv / 128.0
                log_series += second_correction
            if n_terms > 2:
                # Third term for even better accuracy
                third_correction = -v_sq_term * (4*v*v - 9) * (4*v*v - 25) * x_inv * x_inv * x_inv / 3072.0
                log_series += third_correction
            
            result[large_x_mask_full] = np.exp(log_prefactor + log_series)
    
    # Handle overflow/NaN
    overflow_mask = np.isinf(result) | np.isnan(result)
    if np.any(overflow_mask):
        # Fallback to scipy for problematic cases
        result[overflow_mask] = iv(v, x[overflow_mask])
    
    return result


def iv_approximate(v, x, accuracy='medium', fallback_to_scipy=True):
    """
    Hybrid approximation for I_v(x) using optimal method for each range
    
    Automatically selects:
    - Power series for small x (0 ≤ x < 0.1)
    - Chebyshev for medium x (0.1 ≤ x < 10)
    - Asymptotic for large x (x ≥ 10)
    
    Parameters:
    -----------
    v : float
        Order (typically 1/3)
    x : array-like
        Arguments (can be scalar or array)
    accuracy : str
        'low', 'medium', or 'high' (controls tolerance/terms)
    fallback_to_scipy : bool
        If True, fall back to scipy for problematic cases
    
    Returns:
    --------
    result : array
        Approximate I_v(x) values
    """
    from scipy.special import iv
    
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    
    # Determine accuracy parameters
    if accuracy == 'low':
        small_max_terms = 10
        asymp_n_terms = 3
    elif accuracy == 'medium':
        small_max_terms = 20
        asymp_n_terms = 5
    else:  # high
        small_max_terms = 30
        asymp_n_terms = 7
    
    # Small x: power series
    mask_small = (x >= 0) & (x < 0.1)
    if np.any(mask_small):
        try:
            result[mask_small] = iv_power_series(v, x[mask_small], 
                                                  max_terms=small_max_terms)
        except Exception as e:
            if fallback_to_scipy:
                result[mask_small] = iv(v, x[mask_small])
            else:
                raise
    
    # Medium x: Chebyshev
    mask_medium = (x >= 0.1) & (x < 10.0)
    if np.any(mask_medium):
        try:
            result[mask_medium] = iv_chebyshev(v, x[mask_medium])
        except Exception as e:
            if fallback_to_scipy:
                result[mask_medium] = iv(v, x[mask_medium])
            else:
                raise
    
    # Large x: asymptotic
    mask_large = x >= 10.0
    if np.any(mask_large):
        try:
            result[mask_large] = iv_asymptotic(v, x[mask_large], 
                                               n_terms=asymp_n_terms)
        except Exception as e:
            if fallback_to_scipy:
                result[mask_large] = iv(v, x[mask_large])
            else:
                raise
    
    # Handle x = 0 explicitly
    zero_mask = (x == 0)
    result[zero_mask] = 0.0
    
    # Handle negative x (shouldn't happen in FitTeD, but be safe)
    negative_mask = x < 0
    if np.any(negative_mask):
        if fallback_to_scipy:
            result[negative_mask] = iv(v, x[negative_mask])
        else:
            result[negative_mask] = np.nan
    
    return result
