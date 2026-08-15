import numpy as np
import pickle
from .model_base import Model_base
from . import numerical_disc_py as numerical_disc
from ..constants import *
from ..data import *
from scipy.special import iv, hyp2f1, gamma
import warnings
from warnings import warn

# Suppress the same noisy warnings gr_disc.py mutes; the temperature / iv
# expressions legitimately produce overflow / divide-by-zero in regions
# where the Stretched-Mummery profile is identically zero.
warnings.filterwarnings("ignore", message="overflow encountered in exp")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="overflow encountered in divide")

# OPTIMIZATION: Try to import custom iv() approximation (optional)
try:
    from .iv_approximation import iv_approximate
    IV_APPROX_AVAILABLE = True
except ImportError:
    IV_APPROX_AVAILABLE = False


__all__ =  ["GR_disc_plus"]
    
##################################################################################
## GR_disc_plus model -- includes relativistic photon physics
##################################################################################

class GR_disc_plus(Model_base):
    def __init__(self, data=None,
              colour_correction=True,
              rest_frame=True, source_redshift=None,
              decay=True, decay_type='pl',
              rise=False, rise_type='gauss',
              default_N=3000,
              use_iv_approximation=False,
              iv_approximation_accuracy='medium',
              radial_grid_spacing='linear',
              use_dynamic_grid=False,
              delta_r_in=None,
              N_per_time_min=300,
              N_per_time_max=3000,
              weight_by_band=False,
              band_weights=None):

        """
            The GR_disc_plus model. See paper for details of the physics.

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

            Inputs (optimization, ported from GR_disc):
                default_N -- int. Default radial grid size for temperature calculations (default: 3000).
                use_iv_approximation -- boolean. Use custom iv() approximation for speedup (default: False).
                iv_approximation_accuracy -- str. Accuracy level: 'low', 'medium', or 'high' (default: 'medium').
                radial_grid_spacing -- str. 'linear' (default) or 'geometric'. Geometric concentrates
                    points near ISCO where the Stretched-Mummery temperature gradient is steepest;
                    `numerical_disc_model` interpolates with np.interp so it accepts non-uniform grids
                    transparently.
                use_dynamic_grid -- bool. Per-time r grids sized to that time's r_max (default: False).
                delta_r_in -- float or None. Inner-edge resolution (rg) for geometric+dynamic grids.
                N_per_time_min -- int. Minimum N for per-time dynamic grids (default: 300).
                N_per_time_max -- int. Maximum N for per-time dynamic grids (default: 3000).
                weight_by_band -- bool. Weight each band equally instead of each data point equally (default: False).
                    When False (default): each data point contributes equally to the likelihood (standard
                    maximum likelihood). When True: each band's chi-squared contribution is divided by its
                    number of data points, so the per-point average from each band contributes equally.
                    Useful when bands differ in cadence rather than physical importance.
                band_weights -- dict or None. Custom per-band weight multipliers (default: None).
                    Dict mapping band name to weight (e.g. {'Swift XRT': 2.0, 'g.ztf': 1.5}). The band's
                    contribution is multiplied by this weight before being summed into the likelihood.
                    Applied after `weight_by_band` normalization. Bands not present default to weight 1.0.
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
        
        self.rmax_raytrace = 300 #gravitational radii

        # ----------------------------------------------------------------
        # OPTIMIZATION CONFIG (ported from gr_disc.py)
        # ----------------------------------------------------------------
        # Custom iv() Bessel approximation: 3-7x speedup at fixed ν=1/3.
        self.use_iv_approximation = use_iv_approximation if IV_APPROX_AVAILABLE else False
        self.iv_approximation_accuracy = iv_approximation_accuracy
        if use_iv_approximation and not IV_APPROX_AVAILABLE:
            warnings.warn("iv() approximation requested but iv_approximation.py is not "
                          "importable. Falling back to scipy.special.iv().")

        # Radial grid spacing.  'geometric' concentrates points near the
        # inner edge where the Stretched-Mummery profile varies fastest.
        if radial_grid_spacing not in ('linear', 'geometric'):
            raise ValueError("radial_grid_spacing must be 'linear' or 'geometric'")
        self.radial_grid_spacing = radial_grid_spacing

        if radial_grid_spacing == 'linear' and default_N < 1000:
            raise ValueError("default_N must be >= 1000 for linear spacing (accuracy)")
        if default_N < 100:
            raise ValueError("default_N must be >= 100 (minimum reasonable value)")
        self.default_N = int(default_N)

        # Dynamic per-time grids.
        self.use_dynamic_grid = bool(use_dynamic_grid)
        self.delta_r_in = delta_r_in
        self.N_per_time_min = int(N_per_time_min)
        self.N_per_time_max = int(N_per_time_max)

        # Band-weighting options (parity with GR_disc):
        #   weight_by_band -- if True, divide each band's chi-squared by its number of points
        #                     so per-point averages contribute equally across bands.
        #   band_weights   -- optional dict mapping band name -> weight multiplier (default 1.0).
        #                     Applied after the weight_by_band normalization.
        self.weight_by_band = bool(weight_by_band)
        self.band_weights = band_weights
        if self.use_dynamic_grid:
            if radial_grid_spacing == 'linear' and delta_r_in is not None:
                warnings.warn("delta_r_in is only used with geometric spacing, ignoring for linear")
            if delta_r_in is not None:
                if delta_r_in <= 0:
                    raise ValueError(f"delta_r_in must be > 0 (got {delta_r_in})")
                if delta_r_in > 10:
                    warnings.warn(f"delta_r_in={delta_r_in} is very large, may result in very coarse grid")
            if self.N_per_time_min < 50:
                raise ValueError("N_per_time_min must be >= 50")
            if self.N_per_time_max < self.N_per_time_min:
                raise ValueError("N_per_time_max must be >= N_per_time_min")


    # switch_colour_correction, _col_corr, and _ones_like are inherited
    # from Model_base.

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

    ######################################
    ## Physics
    ######################################

    @classmethod
    def get_isco(self, a):
        """
            Returns the ISCO radius as a funciton of black hole spin.
        """
        Z_1 = 1 + (1-a**2)**(1/3) * ((1+a)**(1/3) + (1-a)**(1/3))
        Z_2 = np.sqrt(3*a**2 + Z_1**2)
        return (3 + Z_2 - np.sign(a) * np.sqrt((3-Z_1)*(3 + Z_1 + 2 * Z_2)))

    # ------------------------------------------------------------------
    # Radial-grid helpers (ported from gr_disc.py)
    # ------------------------------------------------------------------

    def _create_radial_grid(self, rI, r_out, N=None, delta_r_in=None):
        """Create a radial grid using the configured spacing method.

        For 'linear' uses np.linspace; for 'geometric' uses np.geomspace
        and (when N is None and delta_r_in is given) sizes N so that the
        first cell width is approximately delta_r_in.
        """
        r_min = rI + 1e-3
        if N is None:
            if self.radial_grid_spacing == 'geometric' and delta_r_in is not None:
                N = self._calculate_N_from_delta_r_in(rI, r_out, delta_r_in)
            else:
                N = self.default_N
        if self.radial_grid_spacing == 'linear':
            return np.linspace(r_min, r_out, N)
        elif self.radial_grid_spacing == 'geometric':
            return np.geomspace(r_min, r_out, N)
        else:
            raise ValueError(f"Unknown radial_grid_spacing: {self.radial_grid_spacing}")

    def _calculate_N_from_delta_r_in(self, rI, r_out, delta_r_in,
                                      N_min=None, N_max=None):
        """Pick N for geometric spacing so r[1] - r[0] ≈ delta_r_in.

        With r_k = r_min · (r_out/r_min)^(k/(N-1)) the first-cell width
        condition gives N ≈ 1 + log(r_out/r_min) / log(1 + delta_r_in/r_min),
        clamped to [N_per_time_min, N_per_time_max].
        """
        r_min = rI + 1e-3
        if N_min is None:
            N_min = self.N_per_time_min
        if N_max is None:
            N_max = self.N_per_time_max
        if r_out <= r_min:
            return N_min
        if delta_r_in <= 0:
            raise ValueError("delta_r_in must be > 0")
        ratio = r_out / r_min
        if ratio <= 1:
            return N_min
        log_ratio = np.log(ratio)
        log_inner = np.log(1 + delta_r_in / r_min)
        if log_inner <= 0:
            return N_min
        N_calc = 1 + int(log_ratio / log_inner)
        return max(N_min, min(N_calc, N_max))

    # ------------------------------------------------------------------
    # Bessel + batched temperature helpers (ported from gr_disc.py)
    # ------------------------------------------------------------------

    def _compute_iv_optimized(self, iv_order, iv_arg, tvsAgrid):
        """iv(ν, iv_arg / tvsAgrid) with optional custom approximation.

        Shapes: iv_arg (N,), tvsAgrid (len(times), 1) → result (len(times), N).
        """
        x = iv_arg / tvsAgrid
        if self.use_iv_approximation:
            try:
                return iv_approximate(
                    iv_order, x,
                    accuracy=self.iv_approximation_accuracy,
                    fallback_to_scipy=True,
                )
            except (ValueError, OverflowError, ImportError) as e:
                warnings.warn(f"iv() approximation failed, using scipy: {e}")
                return iv(iv_order, x)
        else:
            return iv(iv_order, x)

    def _compute_T_batch_times(self, r, taus, log_mh, a_bh, m_disc, r0,
                                tvi, rI, M, Md, rg, alpha):
        """Vectorised Stretched-Mummery T(r, t) for many epochs on a fixed r.

        Computes the time-independent fa, fa0, hyp2f1, gamma blocks ONCE,
        then broadcasts the time-dependent tvsAgrid and a single iv() call.
        Returns shape (len(taus), len(r)).
        """
        taus = np.atleast_1d(taus)
        mu = 0

        # Time-independent quantities
        x = 2 * r / rI
        x0 = 2 * r0 / rI

        R = 2**(alpha - 2) / (alpha * (alpha - 1)) * pi**0.5 * gamma(2 - alpha) / gamma(3/2 - alpha)

        div2x = 2 / x
        fa = (1 - div2x)**0.5 * x**(alpha - 1) / (2 * alpha) * (
            x - 1 / (alpha - 1) * hyp2f1(1, 3/2 - alpha, 2 - alpha, div2x)
        )
        fa += R

        div2x0 = 2 / x0
        fa0 = (1 - div2x0)**0.5 * x0**(alpha - 1) / (2 * alpha) * (
            x0 - 1 / (alpha - 1) * hyp2f1(1, 3/2 - alpha, 2 - alpha, div2x0)
        )
        fa0 += R

        c0 = x0**(-1/8 - 14/8 * mu) * (1 - 2/x0)**(-3/4 + 3/(8*alpha)) * np.sqrt(fa0 * np.exp(1/x0))
        norm = Md / (2 * pi * rI**2 * rg**2) * c0

        w_physical = 2 * np.sqrt(G * M * (r0 * rg)**3) / (((3 - 2*mu)**2.0) * tvi * 60 * 60 * 24)

        # Time-dependent block (vectorised over taus)
        tvsA = 2 * r0**1.5 / ((3 - 2*mu)**2.0) * 2**0.5 * (1 - rI/r0)**1 * (1/rI**1.5) * taus
        tvsAgrid = tvsA[:, np.newaxis]   # (len(taus), 1) → broadcasts to (len(taus), N)

        iv_order = 1 / (4 * alpha)
        iv_arg = fa * (fa0 / 2)
        iv_result = self._compute_iv_optimized(iv_order, iv_arg, tvsAgrid)

        S = (
            fa**0.5 * x**(-(3 + 4*mu + 2*alpha)/4)
            * np.exp(-1/(2*x) - (fa**2 + fa0**2) / (4 * tvsAgrid))
            * 1 / tvsAgrid
            * (1 - div2x)**((10*alpha - 3) / (8*alpha))
            * iv_result * norm
        )

        divr15 = 1 / r**1.5
        T = (3 * (G*M)**0.5 / (4 * O_sb) * rg**-2.5 * r**(-2.5 + mu)
             * w_physical * r0**-mu * S
             * (1 + a_bh*divr15) / (1 - 3/r + 2 * a_bh*divr15)**1.5)**0.25

        # Negative times: profile not yet formed → zero
        negative_mask = taus < 0
        if np.any(negative_mask):
            T[negative_mask, :] = 0
        T[T != T] = 0
        return T

    def get_Temperature(self, times, log_mh, a_bh, m_disc, r0, tvi, t0, N=None):#assumes times in order
        """
            Returns the disc temperature profile T(r, t), given input parameters.

            times = the times (in days) in which the model will return the temperature profile.

            log_mh = the logarithm of the black hole mass in units of solar masses.
            a_bh = the black hole spin (dimensionless -0.999 < a_bh < 0.999).
            m_disc = the disc mass in solar masses.
            r0 = the radial location of the initial disc density spike (in units of gravitational radii).
            tvi = the viscous timescale in days.
            t0 = time before first observation that the disc formed (days).

            N = number of radial grid cells. None → use self.default_N.

            Returns
            -------
            When use_dynamic_grid=False:
                (r, T) with r 1-D shape (N,) and T 2-D shape (len(times), N).
            When use_dynamic_grid=True:
                (r_list, T_list) with one (r_i, T_i) per epoch.
        """
        times = np.atleast_1d(times)
        if N is None:
            N = self.default_N

        mu = 0
        alpha = (3 - 2*mu)/4
        at_these_t_vs = (np.array(times) + t0)/tvi

        if self.use_dynamic_grid:
            # ----------------------------------------------------------
            # Dynamic per-time grids (one (r_i, T_i) per epoch group)
            # ----------------------------------------------------------
            rI = self.get_isco(a_bh)
            M = 10**log_mh * Ms
            Md = m_disc * Ms
            rg = G * M / c**2

            r_max_list = []
            for tau in at_these_t_vs:
                if tau < 0:
                    r_max_list.append(1000)
                else:
                    r_max = max([(int((1 + 5 * tau) * r0) // 10 + 1) * 10, 1000])
                    r_max_list.append(r_max)

            # Group epochs by r_max so we can vectorise the T evaluation
            # within each group while keeping the per-time grid contract.
            r_max_to_indices = {}
            for i, (tau, r_max_i) in enumerate(zip(at_these_t_vs, r_max_list)):
                r_max_to_indices.setdefault(r_max_i, []).append((i, tau))

            r_list = [None] * len(at_these_t_vs)
            T_list = [None] * len(at_these_t_vs)
            for r_max_i, indices_and_taus in r_max_to_indices.items():
                taus_for_rmax = np.array([tau for _, tau in indices_and_taus])
                indices_for_rmax = [i for i, _ in indices_and_taus]
                if r_max_i == 1000 and np.all(taus_for_rmax < 0):
                    r_i = self._create_radial_grid(rI, r_max_i, N=None,
                                                   delta_r_in=self.delta_r_in)
                    for idx, i in enumerate(indices_for_rmax):
                        r_list[i] = r_i
                        T_list[i] = np.zeros(len(r_i))
                    continue
                r_i = self._create_radial_grid(rI, r_max_i, N=None,
                                               delta_r_in=self.delta_r_in)
                T_batch = self._compute_T_batch_times(
                    r_i, taus_for_rmax, log_mh, a_bh, m_disc, r0,
                    tvi, rI, M, Md, rg, alpha,
                )
                for idx, i in enumerate(indices_for_rmax):
                    r_list[i] = r_i
                    T_list[i] = T_batch[idx, :]
            return r_list, T_list

        # --------------------------------------------------------------
        # Static (single-grid) path -- same shape contract as the old
        # implementation but: (a) goes through _create_radial_grid so it
        # honours radial_grid_spacing, (b) uses _compute_T_batch_times so
        # fa/fa0/hyp2f1 are computed once across all epochs.
        # --------------------------------------------------------------
        max_tau = at_these_t_vs[-1]   # times-in-order assumption preserved
        r_out = max([(int((1 + 5 * max_tau) * r0) // 10 + 1) * 10, 1000])

        rI = self.get_isco(a_bh)
        r = self._create_radial_grid(rI, r_out, N)

        M = 10**log_mh * Ms
        Md = m_disc * Ms
        rg = G * M / c**2

        T = self._compute_T_batch_times(
            r, np.asarray(at_these_t_vs), log_mh, a_bh, m_disc, r0,
            tvi, rI, M, Md, rg, alpha,
        )
        return r, T


    def model_X(self, times,
                    log_mh, a_bh, m_disc, r0, tvi, t0, incl, N=None,  El=0.3, Eh=10):
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

        vs = np.geomspace(El*keV_to_Hz, Eh*keV_to_Hz, num=100)
        s = self.model_SEDs(times=times, log_mh=log_mh, a_bh=a_bh,  
                      m_disc=m_disc, r0=r0,  tvi=tvi, t0=t0, incl=incl, vs=vs, N=N, Auv=1)
        
        Lx = ( np.dot(s[:-1].T,(vs[1:] - vs[:-1])/vs[:-1]) ) 
        return Lx

    def model_UV(self, times,
                    log_mh, a_bh, m_disc, r0, tvi, t0, incl, v,  N=None, Auv=1):
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
        l = self.model_SEDs(times=times, log_mh=log_mh, a_bh=a_bh,  
                      m_disc=m_disc, r0=r0,  tvi=tvi, t0=t0, incl=incl, N=N, Auv=Auv, vs=np.array([v]))

        return np.asarray(l[0, :])
    
    def model_SED(self, time,
                    log_mh, a_bh, m_disc, r0, tvi, t0, incl, vs, N=None, Auv=1):
        """
            Returns a model SED at a given time for the input parameters vL_v(v).

            See model_SEDs for parameter docs; this is the single-time wrapper.
        """
        l = self.model_SEDs(times=[time], log_mh=log_mh, a_bh=a_bh,
                             m_disc=m_disc, r0=r0, tvi=tvi, t0=t0,
                             incl=incl, vs=vs, N=N, Auv=Auv)
        return np.asarray(l[:, 0])

    def model_SEDs(self, times,
                    log_mh, a_bh, m_disc, r0, tvi, t0, incl, vs, N=None, Auv=1):
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

            N = number of radial grid cells; None → use self.default_N.

            vs = list of observing frequencies (Hz).

            Returns -- vL_v(vs, times), the disc SED at observed frequencies vs, at times times.
        """
        # Auto-inject r_isco for r-aware colour-correction prescriptions.
        self.fc = self._resolve_fc(r_isco=self.get_isco(a_bh))

        times = np.atleast_1d(times)
        vs = np.atleast_1d(np.asarray(vs, dtype=np.float64))

        DiscR, DiscT = self.get_Temperature(times=times, log_mh=log_mh, a_bh=a_bh,
                                      m_disc=m_disc, r0=r0, tvi=tvi, t0=t0, N=N)

        per_time_grids = isinstance(DiscT, list) or isinstance(DiscR, list)
        return self.model_SEDs_from_T(times, DiscR, DiscT, log_mh, a_bh, incl, vs,
                                       per_time_grids=per_time_grids, Auv=Auv)

    def model_SEDs_from_T(self, times, r, T, log_mh, a_bh, incl, vs,
                           per_time_grids=False, Auv=1,
                           source_type='keplerian'):
        """SED on pre-computed temperature profile(s).

        Groups epochs by radial-grid identity ((len, first, last)) so that
        each unique grid only triggers one kerrgeo trace and one outer-disc
        Planck integration; per-epoch only the temperature differs.

        Parameters
        ----------
        times : array_like, shape (Nt,)
        r : 1-D array OR list of 1-D arrays
            Common-grid (use_dynamic_grid=False) → shape (Nr,).
            Per-time grids (use_dynamic_grid=True) → list of length Nt.
        T : 2-D array OR list of 1-D arrays
            Matching layout: (Nt, Nr) or list of (Nr_i,).
        per_time_grids : bool
            Force the per-time-grid branch.  Auto-detected from r/T types
            when False.
        """
        times = np.atleast_1d(times)
        vs = np.atleast_1d(np.asarray(vs, dtype=np.float64))

        if per_time_grids or (isinstance(r, list) and isinstance(T, list)):
            r_list = r
            T_list = T
        else:
            # Promote the common-grid case to per-time list form so we can
            # share one code path.  The grid-grouping below collapses all
            # epochs back into a single trace + Planck call when r is shared.
            T_arr = np.atleast_2d(T)
            r_list = [r] * T_arr.shape[0]
            T_list = [T_arr[i, :] for i in range(T_arr.shape[0])]

        # Group epochs by radial-grid identity (len, first, last).  Times
        # sharing a grid get one trace + one outer-disc Planck call.
        grid_to_indices = {}
        for i, r_i in enumerate(r_list):
            if len(r_i) > 0:
                grid_key = (len(r_i), float(r_i[0]), float(r_i[-1]))
            else:
                grid_key = (0, 0.0, 0.0)
            grid_to_indices.setdefault(grid_key, []).append(i)

        Nt = len(times)
        LUVs = np.zeros((len(vs), Nt))
        g_z = 1 / (1 + self.source_redshift)

        for grid_key, indices in grid_to_indices.items():
            if not indices:
                continue
            r_i = r_list[indices[0]]
            T_group = np.array([T_list[k] for k in indices])  # (Ngroup, Nr_i)
            # vs are observer-frame Hz; the Spectrum call expects rest-frame
            # frequencies, hence the /g_z.  Rest-frame DiscT in keV.
            DiscT_keV = T_group * kelvin_to_keV
            s = self._get_spectrum_grouped(
                r_i, DiscT_keV, log_mh=log_mh, a_bh=a_bh, incl=incl,
                vs=vs / g_z, source_type=source_type,
            ) * Auv * g_z**3   # shape (len(vs), Ngroup)
            LUVs[:, indices] = s

        return LUVs


    
    def planck(self, v, T):
        """
            The planck function. 

            Returns -- B(v, T). 
        """
        return 2 * h * v**3/c**2 * np.exp(-h*v/(kb * T))/(1-np.exp(-h*v/(kb * T)))
    
    def get_Spectrum(self, DiscR, DiscT, log_mh, a_bh, incl, vs):
        """
            The disc spectrum, for a given disc radius and temperature grid.
            Public API: ``DiscT`` is a length-1 list (or ndarray) of 1-D
            temperature profiles in keV.  Internally delegates to the
            grouped vectorised path.

            Input:
                DiscR, DiscT -- radial and temperature disc grid (DiscT in keV).
                log_mh -- log_10 of black hole mass.
                a_bh -- black hole spin parameter.
                incl -- disc-observer inclination angle (degrees).
                vs -- frequencies to evaluate spectrum at.

            Returns -- vLv(vs).
        """
        DiscR = np.asarray(DiscR, dtype=np.float64)
        # Promote DiscT to 2-D (1, Nr) so the grouped backend handles it.
        DiscT_arr = np.atleast_2d(np.asarray(DiscT, dtype=np.float64))
        return self._get_spectrum_grouped(
            DiscR, DiscT_arr, log_mh=log_mh, a_bh=a_bh, incl=incl,
            vs=np.atleast_1d(np.asarray(vs, dtype=np.float64)),
        )

    def _get_spectrum_grouped(self, DiscR, DiscT_keV, log_mh, a_bh, incl, vs,
                                 source_type='keplerian'):
        """Vectorised disc spectrum for a stack of temperature profiles
        sharing one radial grid.

        Parameters
        ----------
        DiscR : (Nr,) float64
            Radial grid in r_g (linear OR geometric — np.interp handles
            non-uniform spacing inside numerical_disc_model).
        DiscT_keV : (Ng, Nr) float64
            Stack of rest-frame disc temperature profiles in keV.
        log_mh, a_bh, incl : floats
            Black-hole mass / spin / inclination (degrees).
        vs : (Nv,) float64
            Rest-frame frequencies in Hz at which to evaluate vL_v.

        Returns
        -------
        LUVs : (Nv, Ng) float64
        """
        Mbh = 10**log_mh
        Ng = DiscT_keV.shape[0]
        Nv = vs.size
        LUVs = np.zeros((Nv, Ng))

        i_raytrace = DiscR < self.rmax_raytrace
        i_outer = ~i_raytrace
        ri_rt = DiscR[i_raytrace]
        ri_outer = DiscR[i_outer]

        # ------------------------------------------------------------------
        # 1. Inner (ray-traced) disc: r < rmax_raytrace.
        # The new numerical_disc_dNdE_at_vs entry point evaluates the
        # Planck integrand only at the user's frequencies (skipping the
        # inherited 300-bin coarse → 2048-bin fine scaffolding) and
        # batches across all Ng epochs in one call (one cached trace,
        # one mask/colour-correction setup per epoch, shared interp
        # weights for r_em → rdisc).  For typical TDE workloads with
        # Nv ~ 10–24 frequencies this is ~10–30× fewer expm1 calls.
        #
        # Clean νL_ν formula (no fragile bin-width cancellation):
        #     νL_ν = 4 · M_BH² · E_obs² · dN/dE · keV_to_erg
        # ------------------------------------------------------------------
        if ri_rt.size > 0:
            E_obs_keV = vs / keV_to_Hz                                   # (Nv,)
            # Pass the user's colour-correction callable through to
            # numerical_disc_py only if they've explicitly chosen something
            # other than the legacy default (True).  When True, leave it
            # as None so numerical_disc_py uses its bit-identical built-in
            # Done+ 2012 (preserving the regression baseline).
            spec = getattr(self, '_colour_correction_spec', True)
            cc_callable = None if spec is True else self.fc
            dN_dE = numerical_disc.numerical_disc_dNdE_at_vs(
                bh_a=a_bh, rout=self.rmax_raytrace, incl=incl,
                kTdisc_array=DiscT_keV[:, i_raytrace],
                rdisc_array=ri_rt,
                vs_obs_Hz=vs,
                source_type=source_type,
                colour_correction=cc_callable,
            )                                                            # (Ng, Nv)
            nuLnu_inner = (4.0 * Mbh**2 * keV_to_erg
                           * (E_obs_keV ** 2)[None, :] * dN_dE)          # (Ng, Nv)
            LUVs += nuLnu_inner.T                                        # (Nv, Ng)

        # ------------------------------------------------------------------
        # 2. Outer disc: r >= rmax_raytrace, plain colour-corrected Planck
        # in the lab frame (g = 1 by construction at large r).  Vectorised
        # across (Nv, Ng, Nr_outer); honours non-uniform dr.
        # ------------------------------------------------------------------
        if ri_outer.size > 0:
            DiscT_outer_keV = DiscT_keV[:, i_outer]                       # (Ng, Nr_outer)
            DiscT_outer_K = DiscT_outer_keV / kelvin_to_keV               # (Ng, Nr_outer)
            # Broadcast ri_outer (Nr_outer,) along the time axis to (Ng, Nr_outer)
            # so the colour-correction prescription receives matched (T, r).
            ri_outer_b = np.broadcast_to(ri_outer, DiscT_outer_K.shape)
            fcol = self.fc(DiscT_outer_K, ri_outer_b)                     # (Ng, Nr_outer)
            # Cell widths.  Linear: scalar; geometric: per-cell np.diff
            # extended at the outer edge.  Apply the radial-grid mask
            # afterwards so dr aligns with the outer subset.
            if self.radial_grid_spacing == 'linear':
                dr_full = np.full_like(DiscR, DiscR[1] - DiscR[0])
            else:
                dr_full = np.diff(DiscR)
                dr_full = np.concatenate([dr_full, [dr_full[-1]]])
            dr_outer = dr_full[i_outer]                                   # (Nr_outer,)

            # planck on (Nv, Ng, Nr_outer) tensor.
            # vs[:, None, None] / g, with g = 1 here.
            T_arg = fcol * DiscT_outer_K + 1e-6                            # (Ng, Nr_outer)
            B = self.planck(vs[:, None, None],
                            T_arg[None, :, :])                             # (Nv, Ng, Nr_outer)
            dL_dr = (1 / fcol**4)[None, :, :] * B                          # (Nv, Ng, Nr_outer)
            integrand = (
                ri_outer[None, None, :]
                * dL_dr
                * dr_outer[None, None, :]
            )
            outer_contrib = (
                4 * pi * vs[:, None] * np.cos(incl * pi / 180)
                * (Mbh * r_g)**2
                * (2 * pi) * integrand.sum(axis=2)
                * 1e7
            )                                                              # (Nv, Ng)
            LUVs += outer_contrib
        return LUVs

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

        OPTIMIZATION: collects every band's times into a single sorted unique
        array and calls get_Temperature ONCE for the whole MCMC step, then
        indexes per band via searchsorted.  This replaces the per-band
        get_Temperature → get_Spectrum chain in the original implementation.
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

        # ---- 1. Pool every band's times into one sorted unique array ----
        all_times_set = set()
        for band in self.data.bands_UV:
            all_times_set.update(self.data.args_band[band][0])
        for band in self.data.bands_X:
            all_times_set.update(self.data.args_band[band][0])
        for band in self.data.bands_X_upperlim:
            all_times_set.update(self.data.args_band[band][0])
        if not all_times_set:
            return 0.0
        all_times = np.array(sorted(all_times_set))

        # ---- 2. Single get_Temperature call for the whole MCMC step -----
        r_all, T_all = self.get_Temperature(all_times, log_mh, a_bh, m_disc,
                                             r0, tvi, t0, N=None)
        use_per_time_grids = isinstance(T_all, list) or isinstance(r_all, list)

        # Vectorised tolerance-based time → index lookup.
        def find_time_indices(t_band):
            t_arr = np.asarray(t_band)
            idx = np.searchsorted(all_times, t_arr, side='left')
            idx = np.clip(idx, 0, len(all_times) - 1)
            diffs_right = np.abs(all_times[idx] - t_arr)
            has_left = idx > 0
            diffs_left = np.full_like(diffs_right, np.inf)
            if np.any(has_left):
                diffs_left[has_left] = np.abs(all_times[idx[has_left] - 1] - t_arr[has_left])
            use_left = has_left & (diffs_left < diffs_right)
            idx[use_left] = idx[use_left] - 1
            matches = (np.minimum(diffs_left, diffs_right) < 1e-10)
            if not np.all(matches):
                # rare floating-point edge case: fall back to argmin per
                # mismatched row only.
                for i in np.where(~matches)[0]:
                    idx[i] = int(np.argmin(np.abs(all_times - t_arr[i])))
            return idx

        def _slice_band(idx_arr):
            if use_per_time_grids:
                return [r_all[i] for i in idx_arr], [T_all[i] for i in idx_arr]
            else:
                return r_all, T_all[idx_arr, :]

        likelihood = 0
        for band in self.data.bands_UV:
            t_band, lum_band, err_band = self.data.args_band[band]
            v_band = self.data.bands_freq[band]
            s = self.data.global_systematic + self.data.bands_systematic[band]

            idx = find_time_indices(t_band)
            r_b, T_b = _slice_band(idx)
            sed_band = self.model_SEDs_from_T(
                t_band, r_b, T_b, log_mh, a_bh, incl, np.array([v_band]),
                per_time_grids=use_per_time_grids,
            )
            dm = np.asarray(sed_band[0, :])

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

            L_band = dm + em
            diff = lum_band - L_band
            var_band = err_band**2.0 + (s * lum_band)**2.0
            band_contribution = -0.5 * ( (diff**2)/var_band ).sum()
            if self.weight_by_band:
                n_points = len(diff)
                if n_points > 0:
                    band_contribution = band_contribution / n_points
            band_weight = self.band_weights.get(band, 1.0) if self.band_weights else 1.0
            likelihood += band_contribution * band_weight

        for band in self.data.bands_X:
            t_band, lum_band, err_band = self.data.args_band[band]
            v_band = self.data.bands_freq[band]
            s = self.data.global_systematic + self.data.bands_systematic[band]

            idx = find_time_indices(t_band)
            r_b, T_b = _slice_band(idx)
            vs_X = np.geomspace(v_band[0]*keV_to_Hz, v_band[1]*keV_to_Hz, num=100)
            sed_band = self.model_SEDs_from_T(
                t_band, r_b, T_b, log_mh, a_bh, incl, vs_X,
                per_time_grids=use_per_time_grids,
            )
            dm = np.dot(sed_band[:-1].T, (vs_X[1:] - vs_X[:-1]) / vs_X[:-1])

            L_band = dm
            diff = lum_band - L_band
            var_band = err_band**2.0 + (s * lum_band)**2.0
            band_contribution = -0.5 * ( (diff**2)/var_band  ).sum()
            if self.weight_by_band:
                n_points = len(diff)
                if n_points > 0:
                    band_contribution = band_contribution / n_points
            band_weight = self.band_weights.get(band, 1.0) if self.band_weights else 1.0
            likelihood += band_contribution * band_weight

        for band in self.data.bands_X_upperlim:
            t_band, lum_band, N_sig = self.data.args_band[band]
            v_band = self.data.bands_freq[band]

            idx = find_time_indices(t_band)
            r_b, T_b = _slice_band(idx)
            vs_X = np.geomspace(v_band[0]*keV_to_Hz, v_band[1]*keV_to_Hz, num=100)
            sed_band = self.model_SEDs_from_T(
                t_band, r_b, T_b, log_mh, a_bh, incl, vs_X,
                per_time_grids=use_per_time_grids,
            )
            dm = np.dot(sed_band[:-1].T, (vs_X[1:] - vs_X[:-1]) / vs_X[:-1])
            L_band = dm
            band_contribution = self._log_upper_limit_likelihood(L_band, lum_band, N_sig)
            if self.weight_by_band:
                n_points = len(L_band)
                if n_points > 0:
                    band_contribution = band_contribution / n_points
            band_weight = self.band_weights.get(band, 1.0) if self.band_weights else 1.0
            likelihood += band_contribution * band_weight

        if not np.isnan(likelihood):
            return likelihood
        return -np.inf

