import numpy as np
import pickle
from .model_base import Model_base
from ..constants import *
from ..data import *
from scipy.special import iv, hyp2f1, gamma
import warnings
warnings.filterwarnings("ignore", message="overflow encountered in exp")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="overflow encountered in divide")


__all__ =  ["GR_disc"]

#########################################
## GR disc model 
#########################################

class GR_disc(Model_base):
    def __init__(self, data=None, 
              colour_correction=True, 
              rest_frame=True, source_redshift=None, 
              decay=True, decay_type='pl', 
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
        if data is None:
            data = Data_Set(manyTDE_name=None, manyTDE_bands=None, 
                            args_UV=[], bands_UV=[], 
                            args_X=[], bands_X=[], 
                            args_UV_upperlim=[], bands_UV_upperlim=[],
                            args_X_upperlim=[], bands_X_upperlim=[],
                            global_systematic=None
                            )

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
        
        self.switch_colour_correction(colour_correction)
        
        self.data=data
        
        self.default_bounds = {"log_mh"    : (0, 10),          # Note that this is in default units of log_10 Solar masses
                               "a_bh"      : (-0.999, +0.999), # Dimensionless
                               "m_disc"    : (1e-3, np.inf),   # Note that this is in units of Solar masses
                               "r0"        : (1, 10000),       # In rg. Will also will be forced to be larger than the ISCO. 
                               "tvi"       : (1, 1000),        # viscous timescale of disc. (days). 
                               "t0"        : (-100, 365.25),   # time prior to peak at which disc formed (days). 
                               "incl"      : (0, 89),          # disc-observer inclination angle
                               "log_L"     : (0, np.inf),      # luminosity peak of non-disc emission. log_10 erg/s at v0 = 6e14 Hz. 
                               "t_decay"   : (0.1, 1000),      # exponential decay timescale  of non-disc emission. (days). 
                               "log_T"     : (4, 5),           # temperature of initial thermal component. (log10 kelvin)
                               "t_fb"      : (0.1, 1000),      # power-law decay timescale  of non-disc emission. (days). 
                               "p"         : (0, 10),          # power-law decay index  of non-disc emission. (days). 
                               "t_peak"    : (-100, 100),      # time, relative to 0 in the data, of peak emission. (days)
                               "sigma"     : (0, 1000),        # gaussian rise timescale  of non-disc emission. (days). 
                               }

        self.default_key_pars = ["log_mh", "a_bh", "m_disc", "r0", "tvi", "t0", "incl"]

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

    def switch_colour_correction(self, colour_correction):
        """
            Allows for either the Done+ 2012 model of fc (if True, which is default), 
            or no colour-correction (fc = 1). 
        """
        if colour_correction:
            self.fc = self._col_corr
        else:
            self.fc = lambda x: np.ones_like(x)
            
    @staticmethod
    def _col_corr(temp):
        """
            The Done+ 2012 model for the colour correction. 
            See paper for details of the physics. 

            Input -- temp in Kelvin (as a function of radius)
            Returns -- fc(T). 
        """
        i_low = (temp <= 3e4)
        i_mid = (temp < 1e5) & ~i_low
        i_high = ~(i_low | i_mid)
        ans = np.empty_like(temp)
        ans[i_low] = 1.
        ans[i_mid] = np.power(temp[i_mid]/3e4, 0.8333598980732597) # Power from math.log( 11598*72000/1e5, base = 1e5/3e4)
        ans[i_high] = np.power(11598*72000/temp[i_high], 1/9)
        return ans

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

    def get_isco(self, a):
        """
            Returns the ISCO radius as a funciton of black hole spin. 
        """
        Z_1 = 1 + (1-a**2)**(1/3) * ((1+a)**(1/3) + (1-a)**(1/3))
        Z_2 = np.sqrt(3*a**2 + Z_1**2)
        return (3 + Z_2 - np.sign(a) * np.sqrt((3-Z_1)*(3 + Z_1 + 2 * Z_2)))

    def get_Temperature(self, times, log_mh, a_bh, m_disc, r0, tvi, t0, N=30000):#assumes times in order
        """
            Returns the disc temperature profile T(r, t), given input parameters. 

            times = the times (in days) in which the model will return the temperature profile. 

            log_mh = the logarithm of the black hole mass in units of solar masses.
            a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).
            m_disc = the disc mass in solar masses.
            r0 = the radial location of the initial disc density spike (in units of gravitational radii).
            tvi = the viscous timescale in days.
            t0 = time before first observation that the disc formed (days).

            N = the number of radial grid cells for the disc temperature. 

            Returns -- r, T(r, t); disc radial grid and temperature profile. 
        """
        mu = 0
        alpha = (3 - 2*mu)/4

        at_these_t_vs = (np.array(times) + t0)/tvi #need np.array?

        max_tau = at_these_t_vs[-1]
        r_out = max([(int((1 + 5 * max_tau) * r0) // 10 + 1) * 10, 1000])

        rI = self.get_isco(a_bh) #ok cause its just one
        r = np.linspace(rI+1e-3, r_out, N)
        
        M = 10**log_mh * Ms
        Md = m_disc * Ms

        rg = G*M/c**2
    
        x = 2*r/rI ##########
        x0 = 2*r0/rI


        tvsA =  2*r0**1.5/((3 - 2*mu)**2.0) * 2**0.5 * (1-rI/r0)**1*(1/rI**1.5) * at_these_t_vs

        R = 2**(alpha - 2)/(alpha*(alpha-1)) * pi**0.5 * gamma(2-alpha)/gamma(3/2 - alpha) # could move out
        
        div2x=2/x
        fa = (1-div2x)**0.5 * x**(alpha-1) / (2*alpha) * ( x - 1/(alpha-1) * (hyp2f1(1, 3/2 - alpha, 2 - alpha, div2x)) )
        fa += R
        
        div2x0=2/x0
        fa0 = (1-div2x0)**0.5 * x0**(alpha-1) / (2*alpha) * ( x0 - 1/(alpha-1) * (hyp2f1(1, 3/2 - alpha, 2 - alpha, div2x0)) )
        fa0 += R

        c0 = x0**(-1/8 - 14/8*mu) * (1-2/x0)**(-3/4 + 3/(8*alpha)) * np.sqrt(fa0*np.exp(1/x0))
        norm = Md/(2*pi*rI**2*rg**2) * c0
    
        w_physical = 2*np.sqrt(G*M*(r0*rg)**3)/(((3 - 2*mu)**2.0)*tvi*60*60*24)

        tvsAgrid=np.repeat(np.transpose([tvsA]),x.size,axis=1)

        S=fa**0.5 * x**(-(3+4*mu+2*alpha)/4) *np.exp(-1/(2*x)-(fa**2 + fa0**2)/(4*tvsAgrid)) * 1/tvsAgrid * (1-div2x)**( (10*alpha - 3)/(8*alpha) ) * iv(1/(4*alpha), fa*(fa0/2)*1/tvsAgrid) * norm
        divr15=1/r**1.5
        T = (3 * (G*M)**0.5/(4 * O_sb) * rg**-2.5 * r**(-2.5+mu) * w_physical * r0**-mu * S * (1 + a_bh*divr15)/(1 - 3/r + 2 * a_bh*divr15)**1.5)**0.25

        ii_neg = at_these_t_vs<0
        T[ii_neg, :] = np.zeros_like(r)## return zero if times before formation probed. 

        ii_nan = T != T
        T[ii_nan] = 0

        return r, T
    
    def planck(self, v, T):
        """
            The planck function. 

            Returns -- B(v, T). 
        """
        nexp=np.exp(-h*v/(kb * T))
        return 2 * h * v**3/c**2 * nexp/(1-nexp)


    def model_X(self, times, log_mh, a_bh, m_disc, r0, tvi, t0, incl, N=3000, El=0.3, Eh=10):
        """
            Returns an "x-ray" light curve as a function of input parameters L_X(t). 
            
            times = the times (in days) in which the model will return the X-ray light curve. 

            log_mh = the logarithm of the black hole mass in units of solar masses.
            a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).
            m_disc = the disc mass in solar masses.
            r0 = the radial location of the initial disc density spike (in units of gravitational radii).
            tvi = the viscous timescale in days.
            t0 = time before first observation that the disc formed (days).
            incl = disc-observer inclination angle. 

            N = the number of radial grid cells for the disc temperature. 

            El = lower bandpass energy (keV)
            Eh = upper bandpass energy (keV).

            Returns -- L_X(t), the integrated disc luminosity observed from El to Eh. 
        """

        vs = np.geomspace(El*keV_to_Hz, Eh*keV_to_Hz, num=20)

        sed = self.model_SEDs(times, log_mh, a_bh, m_disc, r0, tvi, t0, incl, vs, N=N)
        Lx = ( np.dot(sed[:-1].T,(vs[1:] - vs[:-1])/vs[:-1]) )
            
        return Lx
    

    def model_UV(self, times, log_mh, a_bh, m_disc, r0, tvi, t0, incl, v, N=3000):
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

        l = self.model_SEDs(times, log_mh, a_bh, m_disc, r0, tvi, t0, incl, vs=np.array([v]), N=N)
        return np.asarray(l[0, :])

    def model_SED(self, time, log_mh, a_bh, m_disc, r0, tvi, t0, incl, vs, N=3000):
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

        r, T = self.get_Temperature([time], log_mh, a_bh, m_disc, r0, tvi, t0, N)
        T = T[0]

        vLvs = np.zeros_like(vs)
        rg = 10**log_mh * r_g 

        norm = (rg)**2
        
        i_use = T > 100

        fcol = self.fc(T[i_use])
        g = 1/(1 + self.source_redshift)

        df_dr = 1/fcol**4 * self.planck(vs[:, None]/g, fcol * T[i_use][None, :]) 
        
        dr = np.ones_like(T[i_use]) * (r[1] - r[0])

        vLvs = 4 * pi * vs/g * norm * ( g**3 * 2*pi*r[i_use][None, :] * df_dr * dr[None, :] ).sum(axis=1) * 1e7 * np.cos(incl*pi/180)## W to erg/s

        return vLvs
    
    def model_SEDs(self, times, log_mh, a_bh, m_disc, r0, tvi, t0, incl, vs, N=3000):
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

        r, T = self.get_Temperature(times, log_mh, a_bh, m_disc, r0, tvi, t0, N)
        vLvs = np.zeros( (*vs.shape, *times.shape) )#could move out
        rg = 10**log_mh * r_g 

        norm = (rg)**2

        i_use = T > 100
        fcol = np.ones(T.shape) #could move out, N or something
        fcol[i_use] = self.fc(T[i_use]) #np.where might be faster
        g = 1/(1 + self.source_redshift)

        df_dr = np.zeros( (*vs.shape, *T.shape) ) #could move out
        df_dr[:, i_use] = 1/fcol[i_use]**4 * self.planck(vs[:, None]/g, fcol[None, i_use] * T[None, i_use]) 
        
        dr = (r[1] - r[0])
        
        vLvs = ( 4 * pi * vs[:, None]/g * norm * 
                  g**3 * 2*pi*(r[None, None, :] * df_dr  ).sum(axis=2) * dr
                 * 1e7 * np.cos(incl*pi/180) 
                )   ## W to erg/s
        return vLvs

    def U0(self, r, a):
        ''' The time component of the orbiting fluids 4-velocity.  
            Assumes that r and a are suitably dimensionless 
        '''
        return (1 + a* np.power(r, -3/2))/np.sqrt(1 - 3/r + 2*a*np.power(r, -3/2))

    def Omega_prime(self, r, a, M_bh):
        ''' The radial derivative of the orbiting fluids angular velocity.  
            Assumes that r and a are suitably dimensionless.
            Returns dimensionlfull value.
        '''
        r_g = G*M_bh/c**2
        return - 3/2 * np.sqrt(G*M_bh/r_g**5) * np.power(r, -5/2) / ((1 + a*np.power(r, -3/2))**2)

    def U_phi_prime(self, r, a):
        """U_phi' (r): derivative of the specific angular momentum of a Kerr BH
            returns dimensionless value [dimensions (GM/r)^0.5.]
        """
        numerator = (a + (r**(3/2)))*(r**2 - 6 * r - 3*(a**2) + 8 * a * np.sqrt(r))
        denominator = (2 * (r**4) * ((1 - 3/r + 2*a*np.sqrt(1/(r**3)))**(3/2)))
        return numerator / denominator
    
    def get_Density(self, times, log_mh, a_bh, m_disc, r0, tvi, t0, N=3000):
        """
            Returns the disc surface density profile Sigma(r, t), given input parameters. 

            times = the times (in days) in which the model will return the density profile. 

            log_mh = the logarithm of the black hole mass in units of solar masses.
            a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).
            m_disc = the disc mass in solar masses.
            r0 = the radial location of the initial disc density spike (in units of gravitational radii).
            tvi = the viscous timescale in days.
            t0 = time before first observation that the disc formed (days).

            N = the number of radial grid cells for the disc temperature. 

            Returns -- r, Sigma(r, times); the radial grid and the disc density.  
        """

        mu = 0
        alpha = (3 - 2*mu)/4

        at_these_t_vs = (np.array(times) + t0)/tvi

        max_tau = max(at_these_t_vs)
        r_out = max([(int((1 + 5 * max_tau) * r0) // 10 + 1) * 10, 1000])


        rI = self.get_isco(a_bh)
        r = np.linspace(rI+1e-3, r_out, N)
        
        M = 10**log_mh * Ms
        Md = m_disc * Ms

        rg = G*M/c**2

        x = 2*r/rI
        x0 = 2*r0/rI
        tvsA =  2*r0**1.5/((3 - 2*mu)**2.0) * 2**0.5 * (1-rI/r0)**1*(1/rI**1.5) * np.asarray(at_these_t_vs)
        

        R = 2**(alpha - 2)/(alpha*(alpha-1)) * np.pi**0.5 * gamma(2-alpha)/gamma(3/2 - alpha)
        
        fa = x**alpha/(2*alpha) * (1-2/x)**0.5 - x**(alpha-1)/(2*alpha*(alpha-1)) * (1-2/x)**0.5 * (hyp2f1(1, 3/2 - alpha, 2 - alpha, 2/x))
        fa += R
        fa0 = x0**alpha/(2*alpha) * (1-2/x0)**0.5 - x0**(alpha-1)/(2*alpha*(alpha-1)) * (1-2/x0)**0.5 * (hyp2f1(1, 3/2 - alpha, 2 - alpha, 2/x0))
        fa0 += R
        
        c0 = x0**(-1/8 - 14/8*mu) * (1-2/x0)**(-3/4 + 3/(8*alpha)) * np.sqrt(fa0*np.exp(1/x0))
        norm = Md/(2*np.pi*rI**2*rg**2) * c0

        T = np.zeros(len(r)*len(tvsA)).reshape(len(tvsA), len(r))   
        Ss = np.zeros(len(r)*len(tvsA)).reshape(len(tvsA), len(r))   
        w_physical = 2*np.sqrt(G*M*(r0*rg)**3)/(((3 - 2*mu)**2.0)*tvi*60*60*24)
        r_physical = r * rg

        for pp, tau in enumerate(tvsA):
            if tau<0:
                Ss[pp] = np.zeros_like(r)
            else:
                S = np.sqrt(fa * x**(-alpha) * np.exp(+1/x) * (1-2/x)**((10*alpha - 3)/(4*alpha))) * x**(1/4)/tau * np.exp(-(fa**2 + fa0**2)/(4*tau)) * iv(1/(4*alpha), fa*fa0/(2*tau)) *  np.exp(-1/x) * x**(-mu-1) * norm
                T[pp] = (3 * (G*M)**0.5/(4 * O_sb * r_physical**2.5) * w_physical * (r/r0)**mu * S * (1 + a_bh/r**1.5)/(1 - 3/r + 2 * a_bh/r**1.5)**1.5)**0.25
                Ss[pp] = - 2 * O_sb * T[pp]**4.0 / self.U0(r, a_bh) * 1/self.Omega_prime(r, a_bh, M) * 1/w_physical
        return r, Ss

    def get_Bolometric(self, times, log_mh, a_bh, m_disc, r0, tvi, t0, N=3000):
        """
            Returns the bolometric light curve as a function of input parameters L_bol(t). 
            
            times = the times (in days) in which the model will return the bolometric luminosity. 

            log_mh = the logarithm of the black hole mass in units of solar masses.
            a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).
            m_disc = the disc mass in solar masses.
            r0 = the radial location of the initial disc density spike (in units of gravitational radii).
            tvi = the viscous timescale in days.
            t0 = time before first observation that the disc formed (days).
            incl = disc-observer inclination angle. 

            N = the number of radial grid cells for the disc temperature. 

            Returns -- L_bol(t), the integrated bolometric disc luminosity at times t. 
        """

        r, T = self.get_Temperature(times, log_mh, a_bh, m_disc, r0, tvi, t0, N)
        L_bol = 2 * 2 * np.pi * (r[None, :] * O_sb * T**4.0 * (r[1]-r[0])).sum(axis=1) * (G * 10**log_mh * Ms/c**2)**2.0 * 1e7
        return L_bol
    
    def get_EddingtonRatio(self, times, log_mh, a_bh, m_disc, r0, tvi, t0, N=3000):
        """
            Returns the Eddington luminosity ratio as a function of input parameters L_bol(t)/L_edd. 
            
            times = the times (in days) in which the model will return the Eddington luminosity ratio. 

            log_mh = the logarithm of the black hole mass in units of solar masses.
            a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).
            m_disc = the disc mass in solar masses.
            r0 = the radial location of the initial disc density spike (in units of gravitational radii).
            tvi = the viscous timescale in days.
            t0 = time before first observation that the disc formed (days).
            incl = disc-observer inclination angle. 

            N = the number of radial grid cells for the disc temperature. 

            Returns -- L_bol(t)/L_edd, the disc luminosity eddington ratio at times t. 
        """

        L_bol = self.get_Bolometric(times, log_mh, a_bh, m_disc, r0, tvi, t0, N)
        L_edd = 10**log_mh * 1.26e38 
        return L_bol/L_edd
    
    def get_Mdot(self, times, log_mh, a_bh, m_disc, r0, tvi, t0, N=3000):
        """
            Returns the mass accretion rate as a function of input parameters Mdot(r, t). 
            
            times = the times (in days) in which the model will return the mass accretion rate. 

            log_mh = the logarithm of the black hole mass in units of solar masses.
            a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).
            m_disc = the disc mass in solar masses.
            r0 = the radial location of the initial disc density spike (in units of gravitational radii).
            tvi = the viscous timescale in days.
            t0 = time before first observation that the disc formed (days).
            incl = disc-observer inclination angle. 

            N = the number of radial grid cells for the accretion rate. 

            Returns -- r, dotM(r, times); the radial grid and the disc accretion rate.  
        """

        mu = 0
        alpha = (3 - 2*mu)/4

        at_these_t_vs = (np.array(times) + t0)/tvi

        max_tau = max(at_these_t_vs)
        r_out = max([(int((1 + 5 * max_tau) * r0) // 10 + 1) * 10, 1000])


        rI = self.get_isco(a_bh)
        r = np.linspace(rI+1e-3, r_out, N)
        
        M = 10**log_mh * Ms
        Md = m_disc * Ms

        rg = G*M/c**2

        x = 2*r/rI
        x0 = 2*r0/rI
        tvsA =  2*r0**1.5/((3 - 2*mu)**2.0) * 2**0.5 * (1-rI/r0)**1*(1/rI**1.5) * np.asarray(at_these_t_vs)
        

        R = 2**(alpha - 2)/(alpha*(alpha-1)) * np.pi**0.5 * gamma(2-alpha)/gamma(3/2 - alpha)
        
        fa = x**alpha/(2*alpha) * (1-2/x)**0.5 - x**(alpha-1)/(2*alpha*(alpha-1)) * (1-2/x)**0.5 * (hyp2f1(1, 3/2 - alpha, 2 - alpha, 2/x))
        fa += R
        fa0 = x0**alpha/(2*alpha) * (1-2/x0)**0.5 - x0**(alpha-1)/(2*alpha*(alpha-1)) * (1-2/x0)**0.5 * (hyp2f1(1, 3/2 - alpha, 2 - alpha, 2/x0))
        fa0 += R
        
        c0 = x0**(-1/8 - 14/8*mu) * (1-2/x0)**(-3/4 + 3/(8*alpha)) * np.sqrt(fa0*np.exp(1/x0))
        norm = Md/(2*np.pi*rI**2*rg**2) * c0

        T = np.zeros(len(r)*len(tvsA)).reshape(len(tvsA), len(r))   
        Ss = np.zeros(len(r)*len(tvsA)).reshape(len(tvsA), len(r))   
        y = np.zeros(len(r)*len(tvsA)).reshape(len(tvsA), len(r))   
        Mdot__ = np.zeros(len(r)*len(tvsA)).reshape(len(tvsA), len(r))   
        Mdot = np.zeros((len(r)-1)*len(tvsA)).reshape(len(tvsA), len(r)-1)   
        w_physical = 2*np.sqrt(G*M*(r0*rg)**3)/(((3 - 2*mu)**2.0)*tvi*60*60*24)
        r_physical = r * rg

        for pp, tau in enumerate(tvsA):
            if tau<0:
                Mdot[pp] = np.zeros_like(r)
            else:
                S = np.sqrt(fa * x**(-alpha) * np.exp(+1/x) * (1-2/x)**((10*alpha - 3)/(4*alpha))) * x**(1/4)/tau * np.exp(-(fa**2 + fa0**2)/(4*tau)) * iv(1/(4*alpha), fa*fa0/(2*tau)) *  np.exp(-1/x) * x**(-mu-1) * norm
                T[pp] = (3 * (G*M)**0.5/(4 * O_sb * r_physical**2.5) * w_physical * (r/r0)**mu * S * (1 + a_bh/r**1.5)/(1 - 3/r + 2 * a_bh/r**1.5)**1.5)**0.25
                Ss[pp] = - 2 * O_sb * T[pp]**4.0 / self.U0(r, a_bh) * 1/self.Omega_prime(r, a_bh, M) * 1/w_physical
                y[pp] = r_physical * Ss[pp] * w_physical / self.U0(r, a_bh)
                Mdot__[pp] = - 2*np.pi * self.U0(r, a_bh) * np.gradient(y[pp], r_physical) / (self.U_phi_prime(r, a_bh) * (G * M / rg)**0.5)
                Mdot[pp] = Mdot__[pp][1:]        
        return r[1:], Mdot
    
    def get_EddingtonAccretionRatio(self, times, log_mh, a_bh, m_disc, r0, tvi, t0, N=3000):
        """
            Returns the Eddington accretion ratio as a function of input parameters Mdot(r, t)/M_dot_edd. 
            
            times = the times (in days) in which the model will return the Eddington accretion ratio. 

            log_mh = the logarithm of the black hole mass in units of solar masses.
            a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).
            m_disc = the disc mass in solar masses.
            r0 = the radial location of the initial disc density spike (in units of gravitational radii).
            tvi = the viscous timescale in days.
            t0 = time before first observation that the disc formed (days).
            incl = disc-observer inclination angle. 

            N = the number of radial grid cells for the disc temperature. 

            Returns -- r, dotM(r, times)/dotM_edd; the radial grid and the disc Eddington accretion rate.  
        """

        r, Mdot = self.get_Mdot(times, log_mh, a_bh, m_disc, r0, tvi, t0, N)
        L_edd = 10**log_mh * 1.26e38 * 1e-7 
        eta = 1 - (1 - 2/(3*self.get_isco(a_bh)))**0.5
        Mdot_edd = L_edd/(eta * c**2.0)
        return r, Mdot/Mdot_edd


    def log_likelihood(self, pars):
        '''
        Log likelihood of the data.  If any parameter is impossible, return -np.inf
        p(data | pars). 
        '''
        if not self.rise:
            if self.decay:
                if len(pars) != 10 and self.decay_model == 'exp':
                    raise ValueError("Expected %d parameters, but got %d." % ((10, len(pars)) ))
                if len(pars) != 11 and self.decay_model == 'pl':
                    raise ValueError("Expected %d parameters, but got %d." % ((11, len(pars)) ))
            else:
                if len(pars) != 7:
                    raise ValueError("Expected %d parameters, but got %d." % ((7, len(pars)) ))
        else:
            if self.decay:
                if len(pars) != 12 and self.decay_model == 'exp':
                    raise ValueError("Expected %d parameters, but got %d." % ((12, len(pars)) ))
                if len(pars) != 13 and self.decay_model == 'pl':
                    raise ValueError("Expected %d parameters, but got %d." % ((13, len(pars)) ))
            else:
                if len(pars) != 9:
                    raise ValueError("Expected %d parameters, but got %d." % ((9, len(pars)) ))

        # Get the parameters
        log_mh, a_bh, m_disc, r0, tvi, t0, incl = pars[:7]
        if not self.rise:
            if self.decay:
                if self.decay_type=='exp':
                    log_L, t_decay, log_T  =  pars[7:]
                elif self.decay_type=='pl':
                    log_L, t_fb, p, log_T  =  pars[7:]
        else:    
            if self.decay:
                if self.decay_type=='exp':
                    log_L, t_decay, t_peak, sigma, log_T  =  pars[7:]
                elif self.decay_type=='pl':
                    log_L, t_fb, p, t_peak, sigma, log_T  =  pars[7:]



        if r0 <= self.get_isco(a_bh):
            return -np.inf
        
        
        # Get the model and early model for the appropriate times and frequencies:
        # Loop over all bands:
        likelihood = 0
        for band in self.data.bands_UV:    
            t_band, lum_band, err_band = self.data.args_band[band]
            v_band = self.data.bands_freq[band]
            s = self.data.global_systematic + self.data.bands_systematic[band]

            dm = self.model_UV(t_band, log_mh, a_bh, m_disc, r0, tvi, t0, incl, v_band)

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

        for band in self.data.bands_X:    
            t_band, lum_band, err_band = self.data.args_band[band]
            v_band = self.data.bands_freq[band]
            s = self.data.global_systematic + self.data.bands_systematic[band]

            dm = self.model_X(t_band, log_mh, a_bh, m_disc, r0, tvi, t0, incl, El=v_band[0], Eh=v_band[1])

            # Finish calculating the luminosities from the models:
            L_band = dm 

            # Calculate and add to the likelihood:
            diff = lum_band - L_band
            var_band = err_band**2.0 + (s * lum_band)**2.0
            likelihood += -0.5 * ( (diff**2)/var_band  ).sum()

        for band in self.data.bands_X_upperlim:    
            t_band, lum_band, N_sig = self.data.args_band[band]
            v_band = self.data.bands_freq[band]
            dm = self.model_X(t_band, log_mh, a_bh, m_disc, r0, tvi, t0, incl, El=v_band[0], Eh=v_band[1])

            # Finish calculating the luminosities from the models:
            L_band = dm 

            # Calculate and add to the likelihood:
            likelihood += self._log_upper_limit_likelihood(L_band, lum_band, N_sig)        

        if not np.isnan(likelihood):
            return likelihood
        return -np.inf


