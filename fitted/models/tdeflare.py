import numpy as np
import pickle
from .model_base import Model_base
from ..constants import *
from ..data import *
import warnings
warnings.filterwarnings("ignore", message="overflow encountered in exp")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="overflow encountered in divide")


__all__ =  ["TDEFLARE"]

#########################################
## TDEFLARE model 
#########################################

class TDEFLARE(Model_base):
    def __init__(self, data=None, 
              rest_frame=True, source_redshift=None,  
              rise=False, rise_type='gauss'):
        
        """
            The GR_disc model. See paper for details of the physics. 
            
            Inputs (FitTeD):
                data -- Instance of the FitTeD Data_Set class. If None, then defaults to empty data set. 
            
            Inputs (physics):
                colour_correction -- boolean. Defaults to Done+2012 f_col(T). 
                rest_frame -- boolean. If true assumes data has been corrected to source rest frame. If false, source_redshift must be specified. 
                source_redshift -- z, the disc-observer cosmological redshift. Only needed if rest_frame = False. 

            Inputs (non-disc):
                decay -- boolean, True = include a decaying component, False = no decaying component. 
                decay_type -- either 'pl', 'exp'. Description of decay model, pl = power-law, exp = exponential. See model_base for more. 
                rise -- boolean, True = include a rise component, False = no rise component. 
                rise_type -- only 'gauss' currently supported, a gaussian rise. 
        """
        decay = True
        decay_type = 'exp'
        self.decay=True
        self.decay_type='exp'

        if data is None:
            data = Data_Set(manyTDE_name=None, manyTDE_bands=None, 
                 args_UV=[], bands_UV=[], 
                 args_X=[], bands_X=[], 
                 args_UV_upperlim=[], bands_UV_upperlim=[],
                 args_X_upperlim=[], bands_X_upperlim=[],
                 global_systematic=None)

        if not rest_frame:
            try:
                super().__init__(source_redshift=data.redshift)
            except Exception as e:
                if source_redshift is not None:
                    super().__init__(source_redshift=source_redshift)
                else:
                    raise ValueError()
        else:
            super().__init__(source_redshift=0)
        
        self.data=data
        
        self.default_bounds = { "log_P" : (30, 50), # Plateau luminosity. log_10 erg/s at v0 = 6e14 Hz. 
                            "log_Tp": (3.5, 5.5), # 'Temperature' of plateau. (log10 kelvin).    
                            "log_L"    : (0, np.inf),# luminosity peak of non-disc emission. log_10 erg/s at v0 = 6e14 Hz. 
                            "t_decay" : (0.1, 1000),# exponential decay timescale  of non-disc emission. (days). 
                            "log_T": (4, 5), # temperature of initial thermal component. (log10 kelvin)
                            "t_fb": (0.1, 1000), # power-law decay timescale  of non-disc emission. (days). 
                            "p": (0, 10), # power-law decay index  of non-disc emission. (days). 
                            "t_peak": (-100, 100), # time, relative to 0 in the data, of peak emission. (days)
                            "sigma": (0, 1000) } # gaussian rise timescale  of non-disc emission. (days). 

        self.default_key_pars = ["log_P", "log_Tp"]

        self.decay_model = self.decay_model_no_decay
        if decay:
            if decay_type == 'exp':
                if not rise:
                    self.decay_model = self.early_model_exp
                    self.default_early_pars = ["log_L", "t_decay", "log_T"]
                else:
                    self.decay_model = self.early_model_exp_with_rise
                    self.default_early_pars = ["log_L", "t_decay", "t_peak", "sigma", "log_T"]
                self.decay_type = decay_type
            elif decay_type == 'pl':
                if not rise:
                    self.decay_model = self.early_model_power_law
                    self.default_early_pars = ["log_L", "t_fb", "p", "log_T"]
                else:
                    self.decay_model = self.early_model_power_law_with_rise
                    self.default_early_pars = ["log_L", "t_fb", "p", "t_peak", "sigma", "log_T"]
                self.decay_type = decay_type
        else:
            self.decay_type = None
        self.decay = decay
        
        self.rise_model = self.rise_model_no_rise
        if rise:
            if rise_type == 'gauss':
                self.rise_model = self.rise_model_gauss
                self.rise_type = rise_type
            else:
                raise ValueError()
        self.rise = rise 


    ######################################
    ## Saving and loading 
    ######################################

    def save(self, name):
        """
            Saves current version of the class as name_MODEL.pickle
        """
        file = open(name+'_MODEL.pickle','wb')
        file.write(pickle.dumps(self.__dict__))
        file.close() 

    def load(self, name):
        """
            Loads name_MODEL.pickle 
        """
        ext = ''
        if name[-7:] != '.pickle':
            ext = '.pickle'
            if name[-6:] != "_MODEL":
                name+="_MODEL"
        try:
            file = open(name+ext,'rb')
        except Exception as e:
            print(e)
            return None
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle) 


    def what_pars(self):
        ''' A function you can call which will print the key parameters reequired for the 
            GR_disc model.
        '''
        print()
        print('You are using the GR_disc model class. ')
        print('The light curves produced by the minimal model require 7 key paramaters.')
        print('These parameters are the following: ')
        print('log_mh = the logarithm of the black hole mass in units of solar masses.')
        print('a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).')
        print('m_disc = the disc mass in solar masses. ')
        print('r0 = the radial location of the initial disc density spike (in units of gravitational radii).')
        print('tvi = the viscous timescale in days.')
        print('t0 = time before first observation that the disc formed (days).')
        print('incl = the inclination angle between the disc-plane and observer (degrees).')
        print()     

    ######################################
    ## Physics
    ######################################

    def planck(self, v, T):
        """
            The planck function. 

            Returns -- B(v, T). 
        """
        nexp=np.exp(-h*v/(kb * T))
        return 2 * h * v**3/c**2 * nexp/(1-nexp)

    def model_UV(self, times, t_peak, logP, log_Tp, v):
        """
            Returns a "UV" light curve as a function of input parameters vL_v(t, v). 
            
            times = the times (in days) in which the model will return the UV light curve. 

            log_mh = the logarithm of the black hole mass in units of solar masses.
            a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).
            m_disc = the disc mass in solar masses.
            r0 = the radial location of the initial disc density spike (in units of gravitational radii).
            tvi = the viscous timescale in days.
            t0 = time before first observation that the disc formed (days).
            incl = disc-observer inclination angle. 

            N = the number of radial grid cells for the disc temperature. 

            v = observing frequency (Hz). 

            Returns -- vL_v(t), the disc spectral luminosity observed at v. 
        """

        l = self.model_SEDs(times, t_peak, logP, log_Tp, vs=np.array([v]))
        return np.asarray(l[0, :])

    def model_SED(self, time, t_peak, logP, log_Tp, vs):
        """
            Returns a model SED at a given time for the input parameters vL_v(v). 
            
            time = the time (in days) in which the model will return the SED. 

            log_mh = the logarithm of the black hole mass in units of solar masses.
            a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).
            m_disc = the disc mass in solar masses.
            r0 = the radial location of the initial disc density spike (in units of gravitational radii).
            tvi = the viscous timescale in days.
            t0 = time before first observation that the disc formed (days).
            incl = disc-observer inclination angle. 

            N = the number of radial grid cells for the disc temperature. 

            vs = list of observing frequencies (Hz). 

            Returns -- vL_v(vs), the disc SED at observed frequencies vs. 
        """


        vLvs = 10**logP * self.k_correct(vs, log_Tp) * np.heaviside(time-t_peak,1)

        return vLvs
    
    def model_SEDs(self, times, t_peak, logP, log_Tp, vs):
        """
            Returns model SEDs at given times for the input parameters vL_v(v, t). 
            
            times = the times (in days) in which the model will return the SEDs.

            log_mh = the logarithm of the black hole mass in units of solar masses.
            a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).
            m_disc = the disc mass in solar masses.
            r0 = the radial location of the initial disc density spike (in units of gravitational radii).
            tvi = the viscous timescale in days.
            t0 = time before first observation that the disc formed (days).
            incl = disc-observer inclination angle. 

            N = the number of radial grid cells for the disc temperature. 

            vs = list of observing frequencies (Hz). 

            Returns -- vL_v(vs, times), the disc SED at observed frequencies vs, at times times. 
        """

        times = np.atleast_1d(times)
        vs = np.atleast_1d(vs)

        vLvs = 10**logP * self.k_correct(vs[:, None], log_Tp) * np.heaviside(times[None, :]-t_peak,1)
        return vLvs

    def log_likelihood(self, pars):
        '''
        Log likelihood of the data.  If any parameter is impossible, return -np.inf
        p(data | pars). 
        '''
        if not self.rise:
            if self.decay:
                if len(pars) != 5 and self.decay_model == 'exp':
                    raise ValueError("Expected %d parameters, but got %d." % ((5, len(pars)) ))
                if len(pars) != 11 and self.decay_model == 'pl':## should never hit. 
                    raise ValueError("Expected %d parameters, but got %d." % ((11, len(pars)) ))
            else:
                if len(pars) != 7:## should never hit. 
                    raise ValueError("Expected %d parameters, but got %d." % ((7, len(pars)) ))
        else:
            if self.decay:
                if len(pars) != 7 and self.decay_model == 'exp':
                    raise ValueError("Expected %d parameters, but got %d." % ((7, len(pars)) ))
                if len(pars) != 13 and self.decay_model == 'pl':## should never hit.
                    raise ValueError("Expected %d parameters, but got %d." % ((13, len(pars)) ))
            else:
                if len(pars) != 9:## should never hit. 
                    raise ValueError("Expected %d parameters, but got %d." % ((9, len(pars)) ))

        # Get the parameters
        logP, logTp = pars[:2]
        if not self.rise:
            if self.decay:
                if self.decay_type=='exp':
                    log_L, t_decay, log_T  =  pars[2:]
                elif self.decay_type=='pl':
                    log_L, t_fb, p, log_T  =  pars[2:]
        else:    
            if self.decay:
                if self.decay_type=='exp':
                    log_L, t_decay, t_peak, sigma, log_T  =  pars[2:]
                elif self.decay_type=='pl':
                    log_L, t_fb, p, t_peak, sigma, log_T  =  pars[2:]

        
        # Get the model and early model for the appropriate times and frequencies:
        # Loop over all bands:
        likelihood = 0
        for band in self.data.bands_UV:    
            t_band, lum_band, err_band = self.data.args_band[band]
            v_band = self.data.bands_freq[band]
            s = self.data.global_systematic + self.data.bands_systematic[band]


            if self.rise:
                dm = self.model_UV(t_band, t_peak, logP, logTp, v_band)
            else:
                dm = self.model_UV(t_band, 0, logP, logTp, v_band)

            if not self.rise:
                if self.decay:
                    if self.decay_type == 'exp':
                        em = self.decay_model(t_band, log_L, t_decay, log_T, v_band)
                    elif self.decay_type == 'pl':
                        em = self.decay_model(t_band, log_L, t_fb, p, log_T, v_band)
                else:
                    em = self.decay_model(t_band)
            else:
                if self.decay:
                    if self.decay_type == 'exp':
                        em = self.decay_model(t_band, log_L, t_decay, t_peak, log_T, v_band)
                    elif self.decay_type == 'pl':
                        em = self.decay_model(t_band, log_L, t_fb, p, t_peak, log_T, v_band)
                else:
                    em = self.decay_model(t_band)

                if self.rise_type=='gauss':
                    em += self.rise_model(t_band, log_L, sigma, t_peak, log_T, v_band)
                    

            # Finish calculating the luminosities from the models:
            L_band = dm + em

            # Calculate and add to the likelihood:
            diff = lum_band - L_band
            var_band = err_band**2.0 + (s * lum_band)**2.0
            likelihood += -0.5 * ( (diff**2)/var_band ).sum()

        if not np.isnan(likelihood):
            return likelihood
        return -np.inf


