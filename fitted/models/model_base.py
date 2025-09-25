import numpy as np
from ..constants import *


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
        return 10**log_L * ((t + t_fb)/t_fb)**(-p) * self.k_correct(v, log_T) # This is a power-law decay

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