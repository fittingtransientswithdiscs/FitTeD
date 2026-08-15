import warnings

import numpy as np
from ..constants import *
from . import colour_correction as _cc


__all__ =  ["Model_base"]


#########################
## Models - base
#########################


class Model_base():
    def __init__(self, source_redshift = 0.0):
        self.source_redshift = source_redshift
        self.upper_limit_likelihood = self._log_upper_limit_likelihood
        self.default_key_pars = []
        self.default_early_pars = []

    # ------------------------------------------------------------------
    # Colour-correction plumbing.
    #
    # Subclasses call ``self.switch_colour_correction(spec)`` from their
    # __init__ to set up self.fc.  ``spec`` is anything ``cc.resolve``
    # accepts: bool, str, number, callable, or a ColourCorrection instance.
    #
    # The legacy ``_col_corr`` and ``_ones_like`` static methods are kept
    # below for pickle compatibility — old saved models bind self.fc to
    # those names, so they must remain on the class.  The static methods
    # are now thin shims that delegate to the new module.
    # ------------------------------------------------------------------

    def switch_colour_correction(self, colour_correction):
        """Set ``self.fc`` from a colour-correction spec.

        Accepts the same forms as :func:`fitted.models.colour_correction.resolve`
        (bool / str / number / callable / ColourCorrection instance).

        Backwards-compatible: ``True`` and ``False`` give the same
        Done+ 2012 / unity behaviour as before.

        For r-aware prescriptions (RadialPiecewise etc.), pass them as
        instances; ``r_isco`` will be auto-injected at SED-prediction
        time by the model.
        """
        # Stash the spec so SED methods can re-resolve with the right
        # r_isco when spin is known.
        self._colour_correction_spec = colour_correction
        self.fc = _cc.resolve(colour_correction)

    def _resolve_fc(self, *, r_isco=None):
        """Resolve self._colour_correction_spec for the current spin.

        Called by SED engines that need an r-aware prescription bound to
        a specific r_isco.  Returns a callable instance that can be used
        for the duration of the call.  T-only and constant prescriptions
        are returned as-is from the cached self.fc.
        """
        spec = getattr(self, '_colour_correction_spec', True)
        return _cc.resolve(spec, r_isco=r_isco)

    @staticmethod
    def _col_corr(T, *args):
        """Legacy Done+ 2012 staticmethod, kept for pickle compatibility.

        Old saved models bind ``self.fc = self._col_corr``; unpickling
        restores that binding by name lookup, so this method must stay
        on the class indefinitely.  Accepts (T,) and (T, r) for
        forward compatibility with the new SED engine signature.
        """
        return _cc.Done2012()(np.asarray(T, dtype=float),
                              args[0] if args else None)

    @staticmethod
    def _ones_like(T, *args):
        """Legacy unity staticmethod, kept for pickle compatibility."""
        return _cc.Unity()(np.asarray(T, dtype=float),
                           args[0] if args else None)
            
    def model_X(self, time, *args, **kwargs):
        raise NotImplementedError("This is a base class")
    
    def model_UV(self, time, *args, **kwargs):
        raise NotImplementedError("This is a base class")

    def k_correct(self, v, log_T, v2=6e14):
        T = 10**log_T
        return (v/v2)**4 * (np.exp(h*v2/(kb * T)) - 1)/(np.exp(h*v/(kb * T)) - 1)        

    def early_model_exp(self, t, log_L, tdecay, log_T, v):
        return 10**log_L * np.exp(-(t/tdecay)) * self.k_correct(v, log_T) # This is an exponential decay

    def early_model_power_law(self, t, log_L, t_fb, p, log_T, v):
        # Fallback power law, L ~ ((t + t_fb)/t_fb)**(-p).  It DIVERGES at
        # t = -t_fb (the disruption time) and is undefined below it.
        #
        # That is a property of the model, not a defect to be patched.  If your
        # data extend before the peak, the right model is rise=True, whose decay
        # branch is ((t - t_peak + t_fb)/t_fb)**(-p) evaluated only where
        # t >= t_peak -- so its base is bounded below by 1 and can never be
        # ill-conditioned.  Use that instead.
        #
        # Data earlier than -t_fb give nan here and hence a -inf likelihood,
        # which correctly rejects any t_fb that would place the disruption after
        # an observation.  Do not "fix" that by returning zero: it removes a real
        # constraint on t_fb and lets an unphysical configuration fit quietly.
        # Warn once instead, so the -inf is legible rather than silent.
        x = (t + t_fb) / t_fb
        if not getattr(self, '_pl_domain_warned', False) and np.any(x <= 0):
            self._pl_domain_warned = True
            warnings.warn(
                "early_model_power_law: data extend to t <= -t_fb, where the "
                "fallback power law diverges and is undefined, so the "
                "likelihood is -inf. This model (decay_type='pl', rise=False) "
                "assumes all data postdate the disruption. For a light curve "
                "with pre-peak coverage use rise=True, whose decay branch is "
                "well conditioned everywhere, or cut the data before the flare.",
                stacklevel=2)
        return 10**log_L * (x)**(-p) * self.k_correct(v, log_T) # This is a power-law decay

    def early_model_exp_with_rise(self, t, log_L, tdecay, t_peak, log_T, v): 
        # So it can handle multiple frequencies
        ii_neg = t < t_peak
        l = np.zeros_like(t)
        l[~ii_neg] = 10**log_L * np.exp(-((t[~ii_neg]-t_peak)/tdecay)) * self.k_correct(v, log_T) # This is an exponential decay
        return l

    def early_model_power_law_with_rise(self, t, log_L, t_fb, p, t_peak, log_T, v):
        ii_neg = t < t_peak
        l = np.zeros_like(t)
        l[~ii_neg] = 10**log_L * ((t[~ii_neg] - t_peak + t_fb)/t_fb)**(-p) * self.k_correct(v, log_T) # This is a power-law decay
        return l

    def rise_model_gauss(self, t, log_L, sigma, t_peak, log_T, v):
        ii_neg = t > t_peak
        l = np.zeros_like(t)
        l[~ii_neg] = 10**log_L * np.exp(-(t[~ii_neg]-t_peak)**2.0/(2*sigma**2.0)) * self.k_correct(v, log_T) # This is a guassian rise
        return l

    def rise_model_no_rise(self, t):
        return np.zeros_like(t)

    def decay_model_no_decay(self, t):
        return np.zeros_like(t)

    def log_likelihood(self, pars, *args, **kwargs):
        raise NotImplementedError("This is a base class")
    
    @staticmethod
    def _log_upper_limit_likelihood(model, upper_lims, N_sigma):
        sigma2=(upper_lims/N_sigma)**2 #assuming upperlimits are N-sigma
        return np.where(model - upper_lims >= 0, -0.5 * ( ((model-upper_lims)**2)/sigma2 ), 0.0).sum()