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

# OPTIMIZATION: Try to import custom iv() approximation (optional)
try:
    from .iv_approximation import iv_approximate
    IV_APPROX_AVAILABLE = True
except ImportError:
    IV_APPROX_AVAILABLE = False


__all__ =  ["GR_disc"]

#########################################
## GR disc model 
#########################################

class GR_disc(Model_base):
    def __init__(self, data=None, 
              colour_correction=True, 
              rest_frame=True, source_redshift=None, 
              decay=True, decay_type='pl', 
              rise=False, rise_type='gauss',
              default_N=None,
              use_iv_approximation=True,
              iv_approximation_accuracy='medium',
              radial_grid_spacing='geometric',
              use_dynamic_grid=False,
              delta_r_in=None,
              N_per_time_min=300,
              N_per_time_max=3000,
              weight_by_band=False,
              band_weights=None,
              fit_log_m_disc=True,
              fit_log_r0=True,
              fit_cos_incl=True,
              fit_log_tvi=True):
        
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
            
            Inputs (optimization):
                default_N -- int or None. Radial grid size for temperature calculations.
                    When None (default), resolves to 1000 for geometric spacing and 3000
                    for linear.  Geometric N=1000 is ~3.6x faster and ~3x more accurate
                    than linear N=3000 on a typical TDE light curve.
                use_iv_approximation -- boolean. Use custom iv() approximation for speedup (default: True).
                    Measured at ~1.7x faster with no measurable change to the likelihood
                    (the radial-grid error dominates it).  Requires numba for the full
                    speedup; falls back to numpy, and ultimately scipy, without it.
                    When enabled, provides 3-7x speedup with medium accuracy (relative error < 1e-3).
                iv_approximation_accuracy -- str. Accuracy level for approximation: 'low', 'medium', or 'high' (default: 'medium').
                    'low': ~1e-4 relative error, fastest
                    'medium': ~5e-5 relative error, recommended
                    'high': ~1e-6 relative error, slower
                radial_grid_spacing -- str. Radial grid spacing method: 'linear' (default) or 'geometric' (default: 'linear').
                    'linear': Uses np.linspace() for uniform spacing (original behavior).
                    'geometric': Uses np.geomspace() for logarithmic spacing. May allow smaller N values
                    while maintaining accuracy, especially near the inner edge where gradients are steeper.
                use_dynamic_grid -- bool. Enable dynamic radial grid per time (default: False).
                    When True, each time gets its own optimized r grid based on that time's r_max.
                    This speeds up early times (smaller grids) while maintaining vectorization.
                delta_r_in -- float or None. Inner edge resolution for geometric spacing (default: None).
                    Units: gravitational radii. Only used when radial_grid_spacing='geometric' and 
                    use_dynamic_grid=True. Controls spacing near ISCO: r[1] - r[0] ≈ delta_r_in.
                    When None, uses default_N. Example: delta_r_in=0.1 means ~0.1 rg spacing.
                N_per_time_min -- int. Minimum points for per-time grids (default: 300).
                    Ensures minimum resolution even with large delta_r_in.
                N_per_time_max -- int. Maximum points for per-time grids (default: 3000).
                    Prevents excessive computation time.
                weight_by_band -- bool. Weight each band equally instead of each data point equally (default: False).
                    When False (default): Each data point contributes equally to the likelihood. This is the standard
                    maximum likelihood approach and maximizes information from well-sampled bands.
                    When True: Each band's contribution is normalized by its number of data points, giving equal weight
                    to the average chi-squared per point from each band. This ensures bands with many points don't
                    dominate the fit, which is useful when point density differences arise from observational scheduling
                    rather than physical importance, or when prioritizing consistency across multiple bands.
                    See Statistical Considerations in documentation for trade-offs.
                band_weights -- dict or None. Custom weights for individual bands (default: None).
                    When None (default): All bands have weight 1.0 (identical to current behavior).
                    When specified: Dictionary mapping band names to weight values (e.g., {'Swift XRT': 2.0, 'g.ztf': 1.5}).
                    Each band's contribution is multiplied by its weight before adding to likelihood.
                    Weights are applied after normalization (if weight_by_band=True).
                    Bands not specified in the dictionary get default weight 1.0.
                    Example: band_weights={'Swift XRT': 2.0} gives X-ray data 2x weight in the fit.
                fit_log_m_disc -- bool. Fit for log10(disc mass) instead of linear disc mass (default: False).
                    When False (default): the 3rd key parameter is m_disc (solar masses), with bounds (1e-3, inf).
                    When True: the 3rd key parameter is log_m_disc = log10(m_disc), with bounds (-3, inf).
                    Internally, the model converts to linear m_disc = 10**log_m_disc for all physics calculations.
                fit_log_r0 -- bool. Fit for log10(r0) instead of linear r0 (default: False).
                    When False (default): the 4th key parameter is r0 (gravitational radii), with bounds (1, 10000).
                    When True: the 4th key parameter is log_r0 = log10(r0), with bounds (0, 4).
                    Internally, the model converts to linear r0 = 10**log_r0 for all physics calculations.
                    This works independently of fit_log_m_disc - both can be in log space simultaneously.
                fit_cos_incl -- bool. Fit for cos(inclination) instead of linear inclination (default: False).
                    When False (default): the 7th key parameter is incl (degrees), with bounds (0, 89).
                    When True: the 7th key parameter is cos_incl = cos(incl_degrees), with bounds 
                    (cos(89°), cos(0°)) ≈ (0.017, 1.0). This is useful when the prior on the observing 
                    angle is isotropic, as the probability distribution is uniform in cos(incl) rather than 
                    uniform in incl. Internally, the model converts cos_incl to incl = arccos(cos_incl) * 180/π 
                    (in degrees) for all calculations. This works independently of fit_log_m_disc and fit_log_r0.
                fit_log_tvi -- bool. Fit for log10(tvi) instead of linear tvi (default: False).
                    When False (default): the 5th key parameter is tvi (days), with bounds (1, 1000).
                    When True: the 5th key parameter is log_tvi = log10(tvi), with bounds (0, 3).
                    Internally, the model converts to linear tvi = 10**log_tvi for all physics calculations.
                    This works independently of fit_log_m_disc, fit_log_r0, and fit_cos_incl.
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
        self.fit_log_m_disc = bool(fit_log_m_disc)
        self.fit_log_r0 = bool(fit_log_r0)
        self.fit_cos_incl = bool(fit_cos_incl)
        self.fit_log_tvi = bool(fit_log_tvi)
        
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

        # Optional: fit disc mass in log10-space (log_m_disc) instead of linear m_disc.
        # Internally, all physics uses linear m_disc; conversion happens in log_likelihood()
        # and in convert_parameters() for chain-sampling utilities.
        if self.fit_log_m_disc:
            # Convert bounds for m_disc -> log_m_disc (auto-convert lower bound; upper bound inf stays inf)
            m_lo, m_hi = self.default_bounds.get("m_disc", (1e-3, np.inf))
            self.default_bounds = dict(self.default_bounds)  # make a shallow copy
            self.default_bounds.pop("m_disc", None)
            self.default_bounds["log_m_disc"] = (np.log10(m_lo), np.log10(m_hi) if np.isfinite(m_hi) else np.inf)

            # Replace key parameter name
            self.default_key_pars = ["log_m_disc" if p == "m_disc" else p for p in self.default_key_pars]

        # Optional: fit initial radius in log10-space (log_r0) instead of linear r0.
        # Internally, all physics uses linear r0; conversion happens in log_likelihood()
        # and in convert_parameters() for chain-sampling utilities.
        if self.fit_log_r0:
            # Convert bounds for r0 -> log_r0
            r0_lo, r0_hi = self.default_bounds.get("r0", (1, 10000))
            self.default_bounds = dict(self.default_bounds)  # make a shallow copy if not already
            self.default_bounds.pop("r0", None)
            self.default_bounds["log_r0"] = (np.log10(r0_lo), np.log10(r0_hi) if np.isfinite(r0_hi) else np.inf)

            # Replace key parameter name
            self.default_key_pars = ["log_r0" if p == "r0" else p for p in self.default_key_pars]

        # Optional: fit inclination in cos-space (cos_incl) instead of linear incl.
        # Internally, all physics uses linear incl (degrees); conversion happens in log_likelihood()
        # and in convert_parameters() for chain-sampling utilities.
        if self.fit_cos_incl:
            # Convert bounds for incl -> cos_incl
            incl_lo, incl_hi = self.default_bounds.get("incl", (0, 89))
            self.default_bounds = dict(self.default_bounds)  # make a shallow copy if not already
            self.default_bounds.pop("incl", None)
            # Convert degrees to radians for cosine, then take cosine
            cos_incl_hi = np.cos(incl_lo * np.pi / 180)  # cos(0°) = 1.0
            cos_incl_lo = np.cos(incl_hi * np.pi / 180)  # cos(89°) ≈ 0.017
            self.default_bounds["cos_incl"] = (cos_incl_lo, cos_incl_hi)

            # Replace key parameter name
            self.default_key_pars = ["cos_incl" if p == "incl" else p for p in self.default_key_pars]

        # Optional: fit viscous timescale in log10-space (log_tvi) instead of linear tvi.
        # Internally, all physics uses linear tvi (days); conversion happens in log_likelihood()
        # and in convert_parameters() for chain-sampling utilities.
        if self.fit_log_tvi:
            # Convert bounds for tvi -> log_tvi
            tvi_lo, tvi_hi = self.default_bounds.get("tvi", (1, 1000))
            self.default_bounds = dict(self.default_bounds)  # make a shallow copy if not already
            self.default_bounds.pop("tvi", None)
            self.default_bounds["log_tvi"] = (np.log10(tvi_lo), np.log10(tvi_hi) if np.isfinite(tvi_hi) else np.inf)

            # Replace key parameter name
            self.default_key_pars = ["log_tvi" if p == "tvi" else p for p in self.default_key_pars]

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
        
        # OPTIMIZATION: Custom iv() approximation (optional)
        # Use custom approximation for 3-7x speedup with medium accuracy
        self.use_iv_approximation = use_iv_approximation if IV_APPROX_AVAILABLE else False
        self.iv_approximation_accuracy = iv_approximation_accuracy
        
        if use_iv_approximation and not IV_APPROX_AVAILABLE:
            warnings.warn("iv() approximation requested but not available. Using scipy.special.iv() instead.")
        
        # OPTIMIZATION: Configurable radial grid spacing
        # 'linear': uniform spacing (original behavior, backward compatible)
        # 'geometric': logarithmic spacing (may allow smaller N with same accuracy)
        if radial_grid_spacing not in ['linear', 'geometric']:
            raise ValueError("radial_grid_spacing must be 'linear' or 'geometric'")
        self.radial_grid_spacing = radial_grid_spacing
        
        # OPTIMIZATION: Configurable default N (radial grid size)
        # Default N=3000 matches original log_likelihood() behavior (faster, good accuracy)
        # Original get_Temperature() used N=30000, but log_likelihood() used N=3000
        # Users can override for speed/accuracy tradeoff (higher N = more accurate but slower)
        # Geometric spacing can use smaller N values while maintaining accuracy
        # default_N is resolved from the spacing when not given explicitly.
        # Geometric concentrates points at the inner edge where T(r) varies
        # fastest, so N=1000 geometric is both faster and ~3x more accurate
        # than N=3000 linear; linear still needs 3000 to be usable.
        if default_N is None:
            default_N = 1000 if radial_grid_spacing == 'geometric' else 3000

        if radial_grid_spacing == 'linear' and default_N < 1000:
            raise ValueError("default_N must be >= 1000 for linear spacing (for accuracy)")
        # For geometric spacing, allow lower N values (no minimum restriction)
        if default_N < 100:
            raise ValueError("default_N must be >= 100 (minimum reasonable value)")
        self.default_N = default_N
        
        # OPTIMIZATION: Dynamic radial grid per time
        # Each time gets its own optimized r grid based on that time's r_max
        # Uses loop-based SED computation for efficiency
        self.use_dynamic_grid = use_dynamic_grid
        self.delta_r_in = delta_r_in
        
        # Band weighting option: normalize each band's contribution by number of points
        # When True, each band's average chi-squared per point contributes equally
        # When False (default), each data point contributes equally (standard maximum likelihood)
        self.weight_by_band = weight_by_band
        
        # Per-band weights: custom weights for individual bands
        # Dictionary mapping band names to weight values (e.g., {'Swift XRT': 2.0})
        # Default weight is 1.0 for bands not specified
        # Weights are applied after normalization (if weight_by_band=True)
        self.band_weights = band_weights
        self.N_per_time_min = N_per_time_min
        self.N_per_time_max = N_per_time_max
        
        # Validation for dynamic grid parameters
        if use_dynamic_grid:
            if radial_grid_spacing == 'linear' and delta_r_in is not None:
                warnings.warn("delta_r_in is only used with geometric spacing, ignoring for linear")
            if delta_r_in is not None:
                if delta_r_in <= 0:
                    raise ValueError(f"delta_r_in must be > 0 (got {delta_r_in})")
                if delta_r_in > 10:
                    warnings.warn(f"delta_r_in={delta_r_in} is very large, may result in very coarse grid")
            
            # Validate per-time grid parameters
            if N_per_time_min < 50:
                raise ValueError("N_per_time_min must be >= 50")
            if N_per_time_max < N_per_time_min:
                raise ValueError("N_per_time_max must be >= N_per_time_min")

    # switch_colour_correction, _col_corr, and _ones_like are inherited
    # from Model_base (kept there as the single implementation; the static
    # methods are pickle-compat shims for old saved models).

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
        ''' A function you can call which will print the key parameters required for the
            GR_disc model, *as this particular model instance samples them*.

            The physical content is always the same seven quantities, but the
            variable that is actually sampled depends on the fit_log_* / fit_cos_incl
            flags.  This prints what log_likelihood(), best_fit() and run_chain()
            will expect, in order, so it never disagrees with default_key_pars.
        '''
        meaning = {
            "log_mh":     "the logarithm of the black hole mass in units of solar masses.",
            "a_bh":       "the black hole spin (dimensionless -0.999 < a_bh < 0.999).",
            "m_disc":     "the disc mass in solar masses.",
            "log_m_disc": "log10 of the disc mass in solar masses.",
            "r0":         "the radial location of the initial disc density spike (gravitational radii).",
            "log_r0":     "log10 of the initial disc radius (gravitational radii).",
            "tvi":        "the viscous timescale in days.",
            "log_tvi":    "log10 of the viscous timescale in days.",
            "t0":         "time before first observation that the disc formed (days).",
            "incl":       "the inclination angle between the disc-plane and observer (degrees).",
            "cos_incl":   "the cosine of the disc-observer inclination angle.",
        }
        early_meaning = {
            "log_L":   "log10 of the peak luminosity of the non-disc component (erg/s at 6e14 Hz).",
            "t_fb":    "the fall-back timescale of the non-disc component (days).",
            "t_decay": "the exponential decay timescale of the non-disc component (days).",
            "p":       "the power-law decay index of the non-disc component.",
            "log_T":   "log10 of the temperature of the non-disc component (kelvin).",
            "t_peak":  "the time of peak of the non-disc component (days).",
            "sigma":   "the Gaussian rise timescale of the non-disc component (days).",
        }
        print()
        print('You are using the GR_disc model class.')
        print('This model samples %d key parameters and %d early-time parameters,'
              % (len(self.default_key_pars), len(self.default_early_pars)))
        print('in this order:')
        print()
        for i, name in enumerate(self.default_key_pars):
            print('  [%2d] %-11s = %s' % (i, name, meaning.get(name, '')))
        for j, name in enumerate(self.default_early_pars):
            print('  [%2d] %-11s = %s' % (len(self.default_key_pars) + j, name,
                                          early_meaning.get(name, '')))
        print()
        transformed = [n for n in self.default_key_pars
                       if n in ("log_m_disc", "log_r0", "log_tvi", "cos_incl")]
        if transformed:
            print('Note: %s %s sampled in a transformed space.'
                  % (', '.join(transformed), 'is' if len(transformed) == 1 else 'are'))
            print('Build a parameter vector with model.pack_parameters(...), which takes')
            print('physical values (m_disc in Msun, r0 in rg, tvi in days, incl in degrees)')
            print('and returns the vector in the order above.')
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

    def _create_radial_grid(self, rI, r_out, N=None, delta_r_in=None):
        """
        Create radial grid with specified spacing method (OPTIMIZATION).
        
        Parameters:
        -----------
        rI : float
            Inner radius (ISCO)
        r_out : float
            Outer radius
        N : int or None
            Number of grid points. If None, will be calculated from delta_r_in
            (for geometric spacing) or use default_N
        delta_r_in : float or None
            Inner edge resolution for geometric spacing (only used if N is None)
            
        Returns:
        --------
        r : array
            Radial grid array
        """
        r_min = rI + 1e-3
        
        # Determine N if not provided
        if N is None:
            if self.radial_grid_spacing == 'geometric' and delta_r_in is not None:
                # Calculate N from delta_r_in
                N = self._calculate_N_from_delta_r_in(rI, r_out, delta_r_in)
            else:
                # Use default_N
                N = self.default_N
        
        if self.radial_grid_spacing == 'linear':
            return np.linspace(r_min, r_out, N)
        elif self.radial_grid_spacing == 'geometric':
            return np.geomspace(r_min, r_out, N)
        else:
            raise ValueError(f"Unknown radial_grid_spacing: {self.radial_grid_spacing}")
    
    def _calculate_N_from_delta_r_in(self, rI, r_out, delta_r_in, N_min=None, N_max=None):
        """
        Calculate N for geometric spacing based on delta_r_in requirement.
        
        For geometric spacing, we want the first grid cell to have width ≈ delta_r_in.
        With geometric spacing: r[k] = r_min * (r_out/r_min)^(k/(N-1))
        
        We want: r[1] - r_min ≈ delta_r_in
        Solving: r[1] = r_min * (r_out/r_min)^(1/(N-1))
                 r[1] - r_min ≈ delta_r_in
                 r_min * ((r_out/r_min)^(1/(N-1)) - 1) ≈ delta_r_in
                 (r_out/r_min)^(1/(N-1)) ≈ 1 + delta_r_in/r_min
                 1/(N-1) * log(r_out/r_min) ≈ log(1 + delta_r_in/r_min)
                 N - 1 ≈ log(r_out/r_min) / log(1 + delta_r_in/r_min)
                 N ≈ 1 + log(r_out/r_min) / log(1 + delta_r_in/r_min)
        
        Parameters:
        -----------
        rI : float
            Inner radius (ISCO)
        r_out : float
            Outer radius
        delta_r_in : float
            Desired inner edge resolution (rg)
        N_min : int or None
            Minimum N (uses self.N_per_time_min if None)
        N_max : int or None
            Maximum N (uses self.N_per_time_max if None)
            
        Returns:
        --------
        N : int
            Number of grid points needed
        """
        r_min = rI + 1e-3  # Same as in _create_radial_grid
        
        # Use instance defaults if not provided
        if N_min is None:
            N_min = self.N_per_time_min
        if N_max is None:
            N_max = self.N_per_time_max
        
        # Avoid division by zero or invalid log
        if r_out <= r_min:
            return N_min
        
        if delta_r_in <= 0:
            raise ValueError("delta_r_in must be > 0")
        
        # Calculate N from formula
        ratio = r_out / r_min
        if ratio <= 1:
            return N_min
        
        # N = 1 + log(r_out/r_min) / log(1 + delta_r_in/r_min)
        log_ratio = np.log(ratio)
        log_inner = np.log(1 + delta_r_in / r_min)
        
        if log_inner <= 0:
            # delta_r_in too large relative to r_min, use minimum N
            return N_min
        
        N_calc = 1 + int(log_ratio / log_inner)
        
        # Apply user-configurable bounds
        N = max(N_min, min(N_calc, N_max))
        
        return N
    
    def _compute_T_single_time(self, r, tau, log_mh, a_bh, m_disc, r0, tvi, rI, M, Md, rg, alpha):
        """
        Compute temperature profile for a single time on given r grid.
        
        This is extracted from the main get_Temperature() logic for reuse.
        
        Parameters:
        -----------
        r : array
            Radial grid for this time
        tau : float
            Normalized time (t + t0) / tvi
        log_mh : float
            Logarithm of black hole mass
        a_bh : float
            Black hole spin
        m_disc : float
            Disc mass
        r0 : float
            Initial disc ring radius
        tvi : float
            Viscous timescale
        rI : float
            ISCO radius
        M : float
            Black hole mass (physical units)
        Md : float
            Disc mass (physical units)
        rg : float
            Gravitational radius
        alpha : float
            Viscosity parameter
        
        Returns:
        --------
        T : array
            Temperature profile (1D array for single time)
        """
        mu = 0
        
        x = 2*r/rI
        x0 = 2*r0/rI
        
        tvsA = 2*r0**1.5/((3 - 2*mu)**2.0) * 2**0.5 * (1-rI/r0)**1*(1/rI**1.5) * tau
        
        R = 2**(alpha - 2)/(alpha*(alpha-1)) * pi**0.5 * gamma(2-alpha)/gamma(3/2 - alpha)
        
        div2x = 2/x
        fa = (1-div2x)**0.5 * x**(alpha-1) / (2*alpha) * ( x - 1/(alpha-1) * (hyp2f1(1, 3/2 - alpha, 2 - alpha, div2x)) )
        fa += R
        
        div2x0 = 2/x0
        fa0 = (1-div2x0)**0.5 * x0**(alpha-1) / (2*alpha) * ( x0 - 1/(alpha-1) * (hyp2f1(1, 3/2 - alpha, 2 - alpha, div2x0)) )
        fa0 += R
        
        c0 = x0**(-1/8 - 14/8*mu) * (1-2/x0)**(-3/4 + 3/(8*alpha)) * np.sqrt(fa0*np.exp(1/x0))
        norm = Md/(2*pi*rI**2*rg**2) * c0
        
        w_physical = 2*np.sqrt(G*M*(r0*rg)**3)/(((3 - 2*mu)**2.0)*tvi*60*60*24)
        
        # For single time, tvsAgrid needs shape (1, len(r)) for broadcasting
        # tvsA is a scalar, we need to broadcast it to (1, len(r))
        tvsAgrid = np.array([tvsA])[:, np.newaxis]  # Shape: (1, 1) - will broadcast to (1, len(r))
        
        iv_order = 1/(4*alpha)
        iv_arg = fa*(fa0/2)  # Shape: (len(r),)
        iv_result = self._compute_iv_optimized(iv_order, iv_arg, tvsAgrid)  # Returns (1, len(r))
        
        # All operations broadcast correctly: fa, x, etc. are (len(r),), tvsAgrid is (1, 1) -> (1, len(r))
        S = fa**0.5 * x**(-(3+4*mu+2*alpha)/4) * np.exp(-1/(2*x)-(fa**2 + fa0**2)/(4*tvsAgrid)) * 1/tvsAgrid * (1-div2x)**( (10*alpha - 3)/(8*alpha) ) * iv_result * norm
        # S has shape (1, len(r))
        
        divr15 = 1/r**1.5  # Shape: (len(r),)
        T = (3 * (G*M)**0.5/(4 * O_sb) * rg**-2.5 * r**(-2.5+mu) * w_physical * r0**-mu * S * (1 + a_bh*divr15)/(1 - 3/r + 2 * a_bh*divr15)**1.5)**0.25
        # T has shape (1, len(r))
        
        if tau < 0:
            T = np.zeros((1, len(r)))  # Keep shape (1, len(r))
        
        T[T != T] = 0  # Handle NaN
        
        return T[0, :]  # Extract single time: return shape (len(r),)
    
    def _compute_T_batch_times(self, r, taus, log_mh, a_bh, m_disc, r0, tvi, rI, M, Md, rg, alpha):
        """
        Compute temperature profiles for multiple times on the same r grid (vectorized).
        
        This is optimized to compute fa, fa0, etc. once and then vectorize over times.
        
        Parameters:
        -----------
        r : array
            Radial grid (same for all times)
        taus : array or list
            Normalized times (t + t0) / tvi for multiple times
        log_mh : float
            Logarithm of black hole mass
        a_bh : float
            Black hole spin
        m_disc : float
            Disc mass
        r0 : float
            Initial disc ring radius
        tvi : float
            Viscous timescale
        rI : float
            ISCO radius
        M : float
            Black hole mass (physical units)
        Md : float
            Disc mass (physical units)
        rg : float
            Gravitational radius
        alpha : float
            Viscosity parameter
        
        Returns:
        --------
        T : array, shape (len(taus), len(r))
            Temperature profiles for all times
        """
        taus = np.atleast_1d(taus)
        mu = 0
        
        # Compute time-independent quantities ONCE (these only depend on r, not tau)
        x = 2*r/rI
        x0 = 2*r0/rI
        
        R = 2**(alpha - 2)/(alpha*(alpha-1)) * pi**0.5 * gamma(2-alpha)/gamma(3/2 - alpha)
        
        div2x = 2/x
        fa = (1-div2x)**0.5 * x**(alpha-1) / (2*alpha) * ( x - 1/(alpha-1) * (hyp2f1(1, 3/2 - alpha, 2 - alpha, div2x)) )
        fa += R
        
        div2x0 = 2/x0
        fa0 = (1-div2x0)**0.5 * x0**(alpha-1) / (2*alpha) * ( x0 - 1/(alpha-1) * (hyp2f1(1, 3/2 - alpha, 2 - alpha, div2x0)) )
        fa0 += R
        
        c0 = x0**(-1/8 - 14/8*mu) * (1-2/x0)**(-3/4 + 3/(8*alpha)) * np.sqrt(fa0*np.exp(1/x0))
        norm = Md/(2*pi*rI**2*rg**2) * c0
        
        w_physical = 2*np.sqrt(G*M*(r0*rg)**3)/(((3 - 2*mu)**2.0)*tvi*60*60*24)
        
        # Now compute time-dependent quantities for all times at once (vectorized)
        tvsA = 2*r0**1.5/((3 - 2*mu)**2.0) * 2**0.5 * (1-rI/r0)**1*(1/rI**1.5) * taus  # Shape: (len(taus),)
        
        # Broadcast tvsA for vectorized computation
        tvsAgrid = tvsA[:, np.newaxis]  # Shape: (len(taus), 1) - broadcasts to (len(taus), len(r))
        
        iv_order = 1/(4*alpha)
        iv_arg = fa*(fa0/2)  # Shape: (len(r),)
        iv_result = self._compute_iv_optimized(iv_order, iv_arg, tvsAgrid)  # Returns (len(taus), len(r))
        
        # Vectorized computation for all times at once
        # All operations broadcast: fa, x are (len(r),), tvsAgrid is (len(taus), 1) -> (len(taus), len(r))
        S = fa**0.5 * x**(-(3+4*mu+2*alpha)/4) * np.exp(-1/(2*x)-(fa**2 + fa0**2)/(4*tvsAgrid)) * 1/tvsAgrid * (1-div2x)**( (10*alpha - 3)/(8*alpha) ) * iv_result * norm
        # S has shape (len(taus), len(r))
        
        divr15 = 1/r**1.5  # Shape: (len(r),)
        T = (3 * (G*M)**0.5/(4 * O_sb) * rg**-2.5 * r**(-2.5+mu) * w_physical * r0**-mu * S * (1 + a_bh*divr15)/(1 - 3/r + 2 * a_bh*divr15)**1.5)**0.25
        # T has shape (len(taus), len(r))
        
        # Handle negative times (before disc formation)
        negative_mask = taus < 0
        if np.any(negative_mask):
            T[negative_mask, :] = 0
        
        # Handle NaN
        T[T != T] = 0
        
        return T  # Shape: (len(taus), len(r))

    def _compute_iv_optimized(self, iv_order, iv_arg, tvsAgrid):
        """
        Optimized iv() computation with optional approximation (OPTIMIZATION)
        
        Parameters:
        -----------
        iv_order : float
            Order of modified Bessel function
        iv_arg : array, shape (N,)
            Argument array
        tvsAgrid : array, shape (len(times), N)
            Time-dependent scaling factor grid
        
        Returns:
        --------
        result : array, shape (len(times), N)
            iv() results
        """
        # Compute arguments: iv_arg has shape (N,), tvsAgrid has shape (len(times), N)
        # Division broadcasts: (N,) / (len(times), N) -> (len(times), N)
        x = iv_arg / tvsAgrid  # Shape: (len(times), N)
        
        # Use custom approximation if enabled
        if self.use_iv_approximation:
            try:
                iv_result = iv_approximate(
                    iv_order, 
                    x, 
                    accuracy=self.iv_approximation_accuracy,
                    fallback_to_scipy=True
                )
                return iv_result
            except (ValueError, OverflowError, ImportError) as e:
                # Fallback to scipy for problematic cases
                warnings.warn(f"iv() approximation failed, using scipy: {e}")
                return iv(iv_order, x)
        else:
            # Direct vectorized call (scipy.special is already optimized)
            return iv(iv_order, x)
    
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

            N = the number of radial grid cells for the disc temperature. 
                If None, uses self.default_N (set in __init__(), default: 3000).
                Higher values (e.g., 10000-30000) provide better accuracy but slower computation.

            Returns -- r, T(r, t); disc radial grid and temperature profile. 
            
            When use_dynamic_grid=True:
                Returns r_list, T_list where:
                - r_list: list of per-time radial grids (one per time)
                - T_list: list of per-time temperature profiles (one per time)
                These are used with per_time_grids=True in model_SEDs_from_T() for loop-based SED computation.
            
            When use_dynamic_grid=False:
                Returns r, T where:
                - r: single radial grid (1D array)
                - T: temperature profile (2D array, shape: (len(times), len(r)))
        """
        times = np.atleast_1d(times)
        
        # OPTIMIZATION: Use instance default_N if N not specified
        if N is None:
            N = self.default_N
        
        mu = 0
        alpha = (3 - 2*mu)/4
        at_these_t_vs = (np.array(times) + t0)/tvi
        
        # Check if dynamic grid is enabled
        if self.use_dynamic_grid:
            # ============================================================
            # DYNAMIC GRID: Per-time r grids (no interpolation)
            # ============================================================
            
            rI = self.get_isco(a_bh)
            M = 10**log_mh * Ms
            Md = m_disc * Ms
            rg = G*M/c**2
            
            # Step 1: Calculate r_max for each time
            r_max_list = []
            for tau in at_these_t_vs:
                if tau < 0:
                    r_max_list.append(1000)  # Minimum
                else:
                    r_max = max([(int((1 + 5 * tau) * r0) // 10 + 1) * 10, 1000])
                    r_max_list.append(r_max)
            
            r_out_max = max(r_max_list)  # Maximum across all times
            
            # Step 2: Batch times by r_max to avoid redundant grid creation
            # Group times by their r_max value (grids can be reused, but T must be computed per time)
            r_max_to_indices = {}
            for i, (tau, r_max_i) in enumerate(zip(at_these_t_vs, r_max_list)):
                if r_max_i not in r_max_to_indices:
                    r_max_to_indices[r_max_i] = []
                r_max_to_indices[r_max_i].append((i, tau))
            
            # Step 3: Create grids once per unique r_max and compute T vectorized for all times with same r_max
            r_list = [None] * len(at_these_t_vs)
            T_list = [None] * len(at_these_t_vs)
            
            for r_max_i, indices_and_taus in r_max_to_indices.items():
                # Extract taus and indices for this r_max
                taus_for_rmax = np.array([tau for _, tau in indices_and_taus])
                indices_for_rmax = [i for i, _ in indices_and_taus]
                
                # OPTIMIZATION: Skip grid creation for negative times (r_max=1000, all zeros)
                if r_max_i == 1000 and np.all(taus_for_rmax < 0):
                    # All negative times - just create minimal grid and zeros
                    r_i = self._create_radial_grid(rI, r_max_i, N=None, 
                                                    delta_r_in=self.delta_r_in)
                    for idx, i in enumerate(indices_for_rmax):
                        r_list[i] = r_i
                        T_list[i] = np.zeros(len(r_i))
                    continue
                
                # OPTIMIZATION: Create grid once for this r_max (reused for all times with this r_max)
                r_i = self._create_radial_grid(rI, r_max_i, N=None, 
                                                delta_r_in=self.delta_r_in)
                
                # OPTIMIZATION: Compute T for ALL times with this r_max at once (vectorized)
                # This computes fa, fa0, etc. ONCE and then vectorizes over times
                # Much faster than computing one at a time
                T_batch = self._compute_T_batch_times(
                    r_i, taus_for_rmax, log_mh, a_bh, m_disc, r0, tvi, rI, M, Md, rg, alpha
                )
                
                # Store grid and T for each time
                for idx, i in enumerate(indices_for_rmax):
                    r_list[i] = r_i
                    T_list[i] = T_batch[idx, :]  # Extract single time from batch
            
            # Step 4: Return per-time grids directly (no interpolation)
            # This is faster than interpolating to a common grid and avoids interpolation errors
            return r_list, T_list
            
        else:
            # ============================================================
            # ORIGINAL BEHAVIOR: Single grid for all times
            # ============================================================
            
            max_tau = at_these_t_vs[-1]
            r_out = max([(int((1 + 5 * max_tau) * r0) // 10 + 1) * 10, 1000])

            rI = self.get_isco(a_bh)
            r = self._create_radial_grid(rI, r_out, N)
            
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

            # OPTIMIZATION: Use broadcasting instead of np.repeat() to avoid array copy
            # tvsA has shape (len(times),), we need shape (len(times), N) for broadcasting
            # Instead of: tvsAgrid = np.repeat(np.transpose([tvsA]), x.size, axis=1)
            # Use: tvsAgrid = tvsA[:, np.newaxis] which creates view, not copy
            tvsAgrid = tvsA[:, np.newaxis]  # Shape: (len(times), 1) - broadcasts to (len(times), N)

            # Use custom approximation if enabled, otherwise use scipy
            iv_order = 1/(4*alpha)
            iv_arg = fa*(fa0/2)
            iv_result = self._compute_iv_optimized(iv_order, iv_arg, tvsAgrid)
            
            S=fa**0.5 * x**(-(3+4*mu+2*alpha)/4) *np.exp(-1/(2*x)-(fa**2 + fa0**2)/(4*tvsAgrid)) * 1/tvsAgrid * (1-div2x)**( (10*alpha - 3)/(8*alpha) ) * iv_result * norm
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
        # Auto-inject r_isco for r-aware colour-correction prescriptions.
        self.fc = self._resolve_fc(r_isco=self.get_isco(a_bh))

        r, T = self.get_Temperature([time], log_mh, a_bh, m_disc, r0, tvi, t0, N)
        T = T[0]

        vLvs = np.zeros_like(vs)
        rg = 10**log_mh * r_g 

        norm = (rg)**2
        
        i_use = T > 100

        # T and r are both 1-D and the same length here (single epoch).
        fcol = self.fc(T[i_use], r[i_use])
        g = 1/(1 + self.source_redshift)

        df_dr = 1/fcol**4 * self.planck(vs[:, None]/g, fcol * T[i_use][None, :])
        
        # OPTIMIZATION: Handle non-uniform spacing (geometric grid)
        # For linear spacing, dr is constant; for geometric, use actual cell widths
        if self.radial_grid_spacing == 'linear':
            dr = np.ones_like(T[i_use]) * (r[1] - r[0])
        else:  # geometric
            # Compute cell widths: for each point, use the width of the cell it represents
            # Use forward differences and extend last cell
            dr_full = np.diff(r)  # N-1 elements
            dr_full = np.concatenate([dr_full, [dr_full[-1]]])  # Extend to N elements
            dr = dr_full[i_use]

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
        # Auto-inject r_isco for r-aware colour-correction prescriptions.
        self.fc = self._resolve_fc(r_isco=self.get_isco(a_bh))

        times = np.atleast_1d(times)
        vs = np.atleast_1d(vs)

        r, T = self.get_Temperature(times, log_mh, a_bh, m_disc, r0, tvi, t0, N)
        
        # Check if we got lists (dynamic grid mode) - if so, use model_SEDs_from_T
        if isinstance(T, list) or isinstance(r, list):
            # Dynamic grid mode: use optimized loop-based computation
            return self.model_SEDs_from_T(times, r, T, log_mh, incl, vs, per_time_grids=True)
        
        # Standard mode: continue with original logic
        vLvs = np.zeros( (*vs.shape, *times.shape) )#could move out
        rg = 10**log_mh * r_g 

        norm = (rg)**2

        i_use = T > 100
        fcol = np.ones(T.shape) #could move out, N or something
        # T is (Nt, Nr); broadcast r along the time axis so (T, r) align.
        r_b = np.broadcast_to(r, T.shape)
        fcol[i_use] = self.fc(T[i_use], r_b[i_use])
        g = 1/(1 + self.source_redshift)

        df_dr = np.zeros( (*vs.shape, *T.shape) ) #could move out
        df_dr[:, i_use] = 1/fcol[i_use]**4 * self.planck(vs[:, None]/g, fcol[None, i_use] * T[None, i_use]) 
        
        # OPTIMIZATION: Handle non-uniform spacing (geometric grid)
        # For linear spacing, dr is constant; for geometric, multiply element-wise before summing
        if self.radial_grid_spacing == 'linear':
            dr = (r[1] - r[0])
            vLvs = ( 4 * pi * vs[:, None]/g * norm * 
                      g**3 * 2*pi*(r[None, None, :] * df_dr  ).sum(axis=2) * dr
                     * 1e7 * np.cos(incl*pi/180) 
                    )   ## W to erg/s
        else:  # geometric
            # Compute cell widths: for each point, use the width of the cell it represents
            # Use forward differences and extend last cell
            dr = np.diff(r)  # N-1 elements: widths between consecutive points
            dr = np.concatenate([dr, [dr[-1]]])  # Extend to N elements (last cell same width)
            # Multiply each cell by its width before summing
            vLvs = ( 4 * pi * vs[:, None]/g * norm * 
                      g**3 * 2*pi*(r[None, None, :] * df_dr * dr[None, None, :]).sum(axis=2)
                     * 1e7 * np.cos(incl*pi/180) 
                    )   ## W to erg/s
        return vLvs

    def model_SEDs_from_T(self, times, r, T, log_mh, incl, vs, per_time_grids=False):
        """
            Returns model SEDs at given times using pre-computed temperature profiles.
            This is an optimized version that avoids recalculating temperature.
            
            times = the times (in days) at which to return the SEDs.
            r = pre-computed radial grid (from get_Temperature) OR list of per-time grids.
            T = pre-computed temperature profile T(r, times) with shape (len(times), len(r))
                OR list of per-time temperature profiles.
            log_mh = the logarithm of the black hole mass in units of solar masses.
            incl = disc-observer inclination angle.
            vs = list of observing frequencies (Hz).
            per_time_grids = if True, r and T are lists of per-time arrays (for testing).
            
            Returns -- vL_v(vs, times), the disc SED at observed frequencies vs, at times times.
        """
        times = np.atleast_1d(times)
        vs = np.atleast_1d(vs)
        
        # Auto-detect if r and T are lists (per-time grids) or arrays (common grid)
        # This allows backward compatibility and handles edge cases
        if per_time_grids or (isinstance(r, list) and isinstance(T, list)):
            per_time_grids = True
        else:
            per_time_grids = False
        
        if per_time_grids:
            # r and T are lists of arrays (per-time grids)
            # OPTIMIZATION: Group times by radial grid to enable vectorization
            vLvs = np.zeros((len(vs), len(times)))
            rg = 10**log_mh * r_g
            norm = (rg)**2
            g = 1/(1 + self.source_redshift)
            
            # Step 1: Group times by their radial grid (times with same r can be vectorized)
            # Use grid identity: compare grid lengths and first/last values for fast grouping
            grid_to_indices = {}
            for i, r_i in enumerate(r):
                # Create a simple hash key: (length, first_val, last_val)
                # This is fast and works for most cases (grids with same r_max will match)
                if len(r_i) > 0:
                    grid_key = (len(r_i), r_i[0], r_i[-1])
                else:
                    grid_key = (0, 0, 0)
                
                if grid_key not in grid_to_indices:
                    grid_to_indices[grid_key] = []
                grid_to_indices[grid_key].append(i)
            
            # Step 2: Process each group of times with the same radial grid
            for grid_key, indices in grid_to_indices.items():
                if len(indices) == 0:
                    continue
                
                # Get the radial grid (all times in this group share the same r)
                r_i = r[indices[0]]
                
                # Stack all T profiles for this group into a 2D array
                # Shape: (num_times_in_group, len(r_i))
                T_group = np.array([T[i] for i in indices])
                
                # Vectorized computation for all times with this grid
                i_use = T_group > 100  # Shape: (num_times, len(r_i))
                fcol = np.ones_like(T_group)
                # T_group is (Ng, len(r_i)); broadcast r_i along the group axis.
                r_i_b = np.broadcast_to(r_i, T_group.shape)
                fcol[i_use] = self.fc(T_group[i_use], r_i_b[i_use])
                
                # Compute df_dr for all times in this group (vectorized)
                # Shape: (len(vs), num_times_in_group, len(r_i))
                df_dr = np.zeros((len(vs), len(indices), len(r_i)))
                
                # Vectorize over times: compute planck for all times at once where possible
                # For each time in the group, compute df_dr
                for j, idx in enumerate(indices):
                    T_j = T_group[j, :]
                    i_use_j = i_use[j, :]
                    fcol_j = fcol[j, :]
                    
                    if np.any(i_use_j):
                        # Vectorized over vs and r for this time
                        df_dr[:, j, i_use_j] = 1/fcol_j[i_use_j]**4 * self.planck(
                            vs[:, None]/g, fcol_j[i_use_j] * T_j[i_use_j]
                        )
                
                # Compute dr once for this grid (shared by all times in group)
                if self.radial_grid_spacing == 'geometric':
                    dr = np.diff(r_i)
                    dr = np.concatenate([dr, [dr[-1]]])
                    # Vectorized computation for all times in this group
                    # r_i[None, None, :] shape: (1, 1, len(r_i))
                    # df_dr shape: (len(vs), num_times, len(r_i))
                    # dr[None, None, :] shape: (1, 1, len(r_i))
                    # Result after sum(axis=2): (len(vs), num_times)
                    vLvs_group = (4 * pi * vs[:, None]/g * norm * g**3 * 
                                   2*pi*(r_i[None, None, :] * df_dr * dr[None, None, :]).sum(axis=2) * 
                                   1e7 * np.cos(incl*pi/180))
                else:  # linear
                    dr = (r_i[1] - r_i[0])
                    # Vectorized computation for all times in this group
                    vLvs_group = (4 * pi * vs[:, None]/g * norm * g**3 * 
                                   2*pi*(r_i[None, None, :] * df_dr).sum(axis=2) * dr * 
                                   1e7 * np.cos(incl*pi/180))
                
                # Store results in correct positions
                vLvs[:, indices] = vLvs_group
                
            
            return vLvs
        else:
            # Original vectorized implementation (common grid)
            # Ensure T has the right shape: (len(times), len(r))
            if T.ndim == 1:
                T = T[None, :]  # Add time dimension if needed
            
            vLvs = np.zeros((*vs.shape, *times.shape))
            rg = 10**log_mh * r_g
            
            norm = (rg)**2
            
            i_use = T > 100
            fcol = np.ones(T.shape)
            # T is (Nt, Nr); broadcast r along the time axis so (T, r) align.
            r_b = np.broadcast_to(r, T.shape)
            fcol[i_use] = self.fc(T[i_use], r_b[i_use])
            g = 1/(1 + self.source_redshift)

            df_dr = np.zeros((*vs.shape, *T.shape))
            df_dr[:, i_use] = 1/fcol[i_use]**4 * self.planck(vs[:, None]/g, fcol[None, i_use] * T[None, i_use])
            
            # OPTIMIZATION: Handle non-uniform spacing (geometric grid)
            # For linear spacing, dr is constant; for geometric, multiply element-wise before summing
            if self.radial_grid_spacing == 'linear':
                dr = (r[1] - r[0])
                vLvs = (4 * pi * vs[:, None]/g * norm * 
                        g**3 * 2*pi*(r[None, None, :] * df_dr).sum(axis=2) * dr
                        * 1e7 * np.cos(incl*pi/180))
            else:  # geometric
                # Compute cell widths: for each point, use the width of the cell it represents
                # Use forward differences and extend last cell
                dr = np.diff(r)  # N-1 elements: widths between consecutive points
                dr = np.concatenate([dr, [dr[-1]]])  # Extend to N elements (last cell same width)
                # Multiply each cell by its width before summing
                vLvs = (4 * pi * vs[:, None]/g * norm * 
                        g**3 * 2*pi*(r[None, None, :] * df_dr * dr[None, None, :]).sum(axis=2)
                        * 1e7 * np.cos(incl*pi/180))
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
        r = self._create_radial_grid(rI, r_out, N)
        
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
        
        # Handle dynamic grid case: get_Temperature returns lists instead of arrays
        if self.use_dynamic_grid:
            # r and T are lists, compute L_bol for each time separately
            L_bol = np.zeros(len(times))
            for i, (r_i, T_i) in enumerate(zip(r, T)):
                r_i = np.asarray(r_i)
                T_i = np.asarray(T_i)
                if self.radial_grid_spacing == 'linear':
                    dr = (r_i[1] - r_i[0])
                    L_bol[i] = 2 * 2 * np.pi * (r_i * O_sb * T_i**4.0 * dr).sum() * (G * 10**log_mh * Ms/c**2)**2.0 * 1e7
                else:  # geometric
                    dr = np.diff(r_i)
                    dr = np.concatenate([dr, [dr[-1]]])  # Extend to N elements
                    L_bol[i] = 2 * 2 * np.pi * (r_i * O_sb * T_i**4.0 * dr).sum() * (G * 10**log_mh * Ms/c**2)**2.0 * 1e7
            return L_bol
        
        # Original behavior: r and T are arrays
        # OPTIMIZATION: Handle non-uniform spacing (geometric grid)
        # For linear spacing, dr is constant; for geometric, multiply element-wise before summing
        if self.radial_grid_spacing == 'linear':
            dr = (r[1] - r[0])
            L_bol = 2 * 2 * np.pi * (r[None, :] * O_sb * T**4.0 * dr).sum(axis=1) * (G * 10**log_mh * Ms/c**2)**2.0 * 1e7
        else:  # geometric
            # Compute cell widths: for each point, use the width of the cell it represents
            # Use forward differences and extend last cell
            dr = np.diff(r)  # N-1 elements: widths between consecutive points
            dr = np.concatenate([dr, [dr[-1]]])  # Extend to N elements (last cell same width)
            L_bol = 2 * 2 * np.pi * (r[None, :] * O_sb * T**4.0 * dr[None, :]).sum(axis=1) * (G * 10**log_mh * Ms/c**2)**2.0 * 1e7
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
        r = self._create_radial_grid(rI, r_out, N)
        
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
                Mdot[pp] = np.zeros(len(r)-1)  # Mdot has shape (len(r)-1,), not len(r)
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

    def convert_parameters(self, pars, param_names=None):
        """
        Convert a parameter vector from chain/user format to model-evaluation format.

        This exists to support sampling from MCMC chains that may have been fit with either
        `m_disc` (linear) or `log_m_disc` (log10) as the disc-mass parameter, and/or
        `r0` (linear) or `log_r0` (log10) as the initial radius parameter, and/or
        `tvi` (linear) or `log_tvi` (log10) as the viscous timescale parameter, and/or
        `incl` (linear) or `cos_incl` (cosine) as the inclination parameter.

        Parameters
        ----------
        pars : array-like
            Parameter array.
        param_names : list[str] or None
            Names corresponding to `pars`. If provided and contains 'log_m_disc', 'log_r0', 'log_tvi', or 'cos_incl',
            we convert those entries to linear. If not provided, we fall back to model flags
            and assume standard parameter positions (m_disc at index 2, r0 at index 3, tvi at index 4, incl at index 6).

        Returns
        -------
        np.ndarray
            Copy of `pars` with transformed parameters converted to linear (m_disc in solar masses, r0 in gravitational radii, tvi in days, incl in degrees).
        """
        pars = np.asarray(pars, dtype=float)
        pars_converted = pars.copy()

        # Convert m_disc (index 2) if needed
        needs_m_conversion = False
        m_idx = 2  # default: disc mass is the 3rd parameter in key parameters

        if param_names is not None:
            try:
                if "log_m_disc" in param_names:
                    needs_m_conversion = True
                    m_idx = param_names.index("log_m_disc")
                elif "m_disc" in param_names:
                    needs_m_conversion = False
                    m_idx = param_names.index("m_disc")
                else:
                    # Unknown naming: fall back to model flag + default index
                    needs_m_conversion = bool(self.fit_log_m_disc)
                    m_idx = 2
            except Exception:
                needs_m_conversion = bool(self.fit_log_m_disc)
                m_idx = 2
        else:
            needs_m_conversion = bool(self.fit_log_m_disc)
            m_idx = 2

        if needs_m_conversion and m_idx < len(pars_converted):
            pars_converted[m_idx] = 10**pars_converted[m_idx]

        # Convert r0 (index 3) if needed
        needs_r0_conversion = False
        r0_idx = 3  # default: r0 is the 4th parameter in key parameters

        if param_names is not None:
            try:
                if "log_r0" in param_names:
                    needs_r0_conversion = True
                    r0_idx = param_names.index("log_r0")
                elif "r0" in param_names:
                    needs_r0_conversion = False
                    r0_idx = param_names.index("r0")
                else:
                    # Unknown naming: fall back to model flag + default index
                    needs_r0_conversion = bool(self.fit_log_r0)
                    r0_idx = 3
            except Exception:
                needs_r0_conversion = bool(self.fit_log_r0)
                r0_idx = 3
        else:
            needs_r0_conversion = bool(self.fit_log_r0)
            r0_idx = 3

        if needs_r0_conversion and r0_idx < len(pars_converted):
            pars_converted[r0_idx] = 10**pars_converted[r0_idx]

        # Convert incl (index 6) if needed
        needs_incl_conversion = False
        incl_idx = 6  # default: incl is the 7th parameter in key parameters

        if param_names is not None:
            try:
                if "cos_incl" in param_names:
                    needs_incl_conversion = True
                    incl_idx = param_names.index("cos_incl")
                elif "incl" in param_names:
                    needs_incl_conversion = False
                    incl_idx = param_names.index("incl")
                else:
                    # Unknown naming: fall back to model flag + default index
                    needs_incl_conversion = bool(self.fit_cos_incl)
                    incl_idx = 6
            except Exception:
                needs_incl_conversion = bool(self.fit_cos_incl)
                incl_idx = 6
        else:
            needs_incl_conversion = bool(self.fit_cos_incl)
            incl_idx = 6

        if needs_incl_conversion and incl_idx < len(pars_converted):
            # Convert cos_incl to incl (degrees): incl = arccos(cos_incl) * 180/π
            pars_converted[incl_idx] = np.arccos(pars_converted[incl_idx]) * 180 / np.pi

        # Convert tvi (index 4) if needed
        needs_tvi_conversion = False
        tvi_idx = 4  # default: tvi is the 5th parameter in key parameters

        if param_names is not None:
            try:
                if "log_tvi" in param_names:
                    needs_tvi_conversion = True
                    tvi_idx = param_names.index("log_tvi")
                elif "tvi" in param_names:
                    needs_tvi_conversion = False
                    tvi_idx = param_names.index("tvi")
                else:
                    # Unknown naming: fall back to model flag + default index
                    needs_tvi_conversion = bool(self.fit_log_tvi)
                    tvi_idx = 4
            except Exception:
                needs_tvi_conversion = bool(self.fit_log_tvi)
                tvi_idx = 4
        else:
            needs_tvi_conversion = bool(self.fit_log_tvi)
            tvi_idx = 4

        if needs_tvi_conversion and tvi_idx < len(pars_converted):
            pars_converted[tvi_idx] = 10**pars_converted[tvi_idx]

        return pars_converted


    def pack_parameters(self, log_mh=None, a_bh=None, m_disc=None, r0=None,
                        tvi=None, t0=None, incl=None, **early):
        """
        Build a parameter vector in *this model's* sampling space from physical values.

        This is the inverse of `convert_parameters`.  You give it physical
        quantities -- disc mass in solar masses, radius in gravitational radii,
        viscous time in days, inclination in degrees -- and it returns the vector
        that `log_likelihood`, `best_fit` and `run_chain` actually expect, in the
        order given by `default_key_pars + default_early_pars`.

        The point of it is that the sampling space depends on how the model was
        constructed.  With the v2.0 defaults the model samples `log_m_disc`,
        `log_r0`, `log_tvi` and `cos_incl`, so a hand-written vector containing
        `m_disc=0.05, incl=70` is not merely inaccurate -- `70` is outside the
        `cos_incl` bounds and the prior returns -inf immediately.  Build the
        vector with this method and it is right whatever the flags say.

        Parameters
        ----------
        log_mh : float
            log10 of the black hole mass in solar masses.
        a_bh : float
            Dimensionless black hole spin, -0.999 < a_bh < 0.999.
        m_disc : float
            Disc mass in solar masses (linear, *not* log).
        r0 : float
            Initial disc radius in gravitational radii (linear).
        tvi : float
            Viscous timescale in days (linear).
        t0 : float
            Time before the first observation that the disc formed, in days.
        incl : float
            Disc-observer inclination in degrees (linear, *not* cosine).
        **early
            Early-time parameters by name, e.g. ``log_L=43.3, t_fb=67.0``.
            The names required are in `default_early_pars`; which ones those are
            depends on `decay`, `decay_type` and `rise`.

        Returns
        -------
        np.ndarray
            Parameter vector ordered as `default_key_pars + default_early_pars`.

        Examples
        --------
        >>> pars = model.pack_parameters(log_mh=7.0, a_bh=0.01, m_disc=0.05,
        ...                              r0=30.0, tvi=15.0, t0=-2.0, incl=70.0,
        ...                              log_L=43.3, t_fb=67.0, p=5/3, log_T=4.8)
        >>> model.log_likelihood(pars)
        """
        physical = {"log_mh": log_mh, "a_bh": a_bh, "m_disc": m_disc,
                    "r0": r0, "tvi": tvi, "t0": t0, "incl": incl}

        # Map each sampled name back to the physical quantity it is a function of.
        transform = {
            "log_m_disc": ("m_disc", np.log10),
            "log_r0":     ("r0",     np.log10),
            "log_tvi":    ("tvi",    np.log10),
            "cos_incl":   ("incl",   lambda x: np.cos(np.deg2rad(x))),
        }

        vector = []
        for name in self.default_key_pars:
            source, f = transform.get(name, (name, None))
            value = physical[source]
            if value is None:
                raise ValueError(
                    "pack_parameters: '%s' is required (this model samples '%s')."
                    % (source, name))
            vector.append(float(f(value)) if f is not None else float(value))

        missing = [n for n in self.default_early_pars if n not in early]
        if missing:
            raise ValueError(
                "pack_parameters: missing early-time parameter(s) %s. "
                "This model needs %s." % (missing, self.default_early_pars))
        unexpected = [n for n in early if n not in self.default_early_pars]
        if unexpected:
            raise ValueError(
                "pack_parameters: unexpected parameter(s) %s. "
                "This model's early-time parameters are %s."
                % (unexpected, self.default_early_pars))

        vector += [float(early[n]) for n in self.default_early_pars]
        return np.array(vector)


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
        log_mh, a_bh, m_disc_or_log, r0_or_log, tvi_or_log, t0, incl_or_cos = pars[:7]
        m_disc = (10**m_disc_or_log) if self.fit_log_m_disc else m_disc_or_log
        r0 = (10**r0_or_log) if self.fit_log_r0 else r0_or_log
        tvi = (10**tvi_or_log) if self.fit_log_tvi else tvi_or_log
        incl = (np.arccos(incl_or_cos) * 180 / np.pi) if self.fit_cos_incl else incl_or_cos
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
        
        # OPTIMIZATION: Collect all unique times from all bands and compute temperature once
        all_times_set = set()
        band_times_map = {}  # Map band -> its time array for later indexing
        
        for band in self.data.bands_UV:
            t_band = self.data.args_band[band][0]
            all_times_set.update(t_band)
            band_times_map[band] = t_band
            
        for band in self.data.bands_X:
            t_band = self.data.args_band[band][0]
            all_times_set.update(t_band)
            band_times_map[band] = t_band
            
        for band in self.data.bands_X_upperlim:
            t_band = self.data.args_band[band][0]
            all_times_set.update(t_band)
            band_times_map[band] = t_band
        
        # Convert to sorted array for get_Temperature
        all_times = np.array(sorted(all_times_set))
        
        # Compute temperature once for all unique times
        # Use instance default_N (user-configurable, default: 3000).
        # NOTE: the pre-1.1 per-band code path used N=30000; set default_N=30000
        # in __init__() to reproduce older results.
        r, T_all = self.get_Temperature(all_times, log_mh, a_bh, m_disc, r0, tvi, t0, N=None)
        
        # Check if dynamic grid returns lists (per-time grids) or arrays (common grid)
        use_per_time_grids = self.use_dynamic_grid
        if use_per_time_grids:
            # r and T_all are lists of arrays (one per time)
            # We'll extract the appropriate ones for each band
            pass
        else:
            # r is 1D array, T_all is 2D array (len(times), len(r))
            pass
        
        # Helper function to find indices: match times with tolerance for floating point precision
        def find_time_indices(t_band, all_times):
            """Find indices in all_times that correspond to t_band (OPTIMIZED)"""
            # Convert to arrays for vectorized operations
            t_band_arr = np.asarray(t_band)
            
            # Use searchsorted for efficiency (O(n log m) where n=len(t_band), m=len(all_times))
            # This gives us a good starting point (insertion index)
            indices = np.searchsorted(all_times, t_band_arr, side='left')
            
            # Clamp indices to valid range (handle edge cases where searchsorted returns len(all_times))
            indices = np.clip(indices, 0, len(all_times) - 1)
            
            # Verify matches are within tolerance (handle floating point precision)
            # Vectorized comparison for all values at once - much faster than loop
            diffs_right = np.abs(all_times[indices] - t_band_arr)
            matches_right = diffs_right < 1e-10
            
            # Check left neighbor too (searchsorted might not give closest match)
            # Only check left if index > 0
            has_left = indices > 0
            diffs_left = np.full_like(diffs_right, np.inf)
            if np.any(has_left):
                left_indices = indices[has_left] - 1
                diffs_left[has_left] = np.abs(all_times[left_indices] - t_band_arr[has_left])
            
            # Use whichever neighbor is closer (or exact match)
            use_left = has_left & (diffs_left < diffs_right)
            indices[use_left] = indices[use_left] - 1
            matches = matches_right | (use_left & (diffs_left < 1e-10))
            
            # If many mismatches (>20%), use original approach for all (conservative)
            # This ensures exact same behavior when times don't overlap well
            mismatch_ratio = np.sum(~matches) / len(matches) if len(matches) > 0 else 0
            if mismatch_ratio > 0.2:
                # Too many mismatches - use original approach for all values to ensure correctness
                indices = np.array([np.argmin(np.abs(all_times - t)) for t in t_band_arr])
            elif not np.all(matches):
                # Only use expensive argmin fallback for remaining mismatches (typically rare)
                # OPTIMIZATION: Original code used list comprehension for ALL values (O(n*m))
                # This only uses expensive fallback for mismatches (typically 0-10% of values)
                mismatch_mask = ~matches
                mismatch_indices = np.where(mismatch_mask)[0]
                
                # For remaining mismatches only, use same logic as original (find closest match)
                for i in mismatch_indices:
                    distances = np.abs(all_times - t_band_arr[i])
                    indices[i] = np.argmin(distances)
            
            return indices
        
        # Get the model and early model for the appropriate times and frequencies:
        # Loop over all bands:
        likelihood = 0
        for band in self.data.bands_UV:    
            t_band, lum_band, err_band = self.data.args_band[band]
            v_band = self.data.bands_freq[band]
            s = self.data.global_systematic + self.data.bands_systematic[band]

            # OPTIMIZATION: Use pre-computed temperature instead of recalculating
            # Find indices in all_times that correspond to this band's times
            band_indices = find_time_indices(t_band, all_times)
            
            if use_per_time_grids:
                # Extract per-time grids and temperatures for this band
                r_band = [r[i] for i in band_indices]
                T_band = [T_all[i] for i in band_indices]
            else:
                # Extract temperature slice from 2D array
                T_band = T_all[band_indices, :]
                r_band = r
            
            # Compute SED using pre-computed temperature
            sed_band = self.model_SEDs_from_T(t_band, r_band, T_band, log_mh, incl, np.array([v_band]), per_time_grids=use_per_time_grids)
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
                    

            # Finish calculating the luminosities from the models:
            L_band = dm + em

            # Calculate and add to the likelihood:
            # OPTIMIZATION: Band weighting option - normalize by number of points when enabled
            # When weight_by_band=True: Each band's average chi-squared per point contributes equally
            # This prevents well-sampled bands from dominating the fit
            # When weight_by_band=False: Standard maximum likelihood (each data point weighted equally)
            diff = lum_band - L_band
            var_band = err_band**2.0 + (s * lum_band)**2.0
            band_contribution = -0.5 * ( (diff**2)/var_band ).sum()
            if self.weight_by_band:
                n_points = len(diff)
                if n_points > 0:
                    band_contribution = band_contribution / n_points
            # Apply per-band weight (default 1.0 if not specified)
            # Weights are applied after normalization (if weight_by_band=True)
            band_weight = self.band_weights.get(band, 1.0) if self.band_weights else 1.0
            likelihood += band_contribution * band_weight

        for band in self.data.bands_X:    
            t_band, lum_band, err_band = self.data.args_band[band]
            v_band = self.data.bands_freq[band]
            s = self.data.global_systematic + self.data.bands_systematic[band]

            # OPTIMIZATION: Use pre-computed temperature instead of recalculating
            # Find indices in all_times that correspond to this band's times
            band_indices = find_time_indices(t_band, all_times)
            
            if use_per_time_grids:
                # Extract per-time grids and temperatures for this band
                r_band = [r[i] for i in band_indices]
                T_band = [T_all[i] for i in band_indices]
            else:
                # Extract temperature slice from 2D array
                T_band = T_all[band_indices, :]
                r_band = r
            
            # Compute X-ray model using pre-computed temperature
            # X-ray model integrates over frequency range, so we compute SEDs at multiple frequencies
            vs = np.geomspace(v_band[0]*keV_to_Hz, v_band[1]*keV_to_Hz, num=20)
            sed_band = self.model_SEDs_from_T(t_band, r_band, T_band, log_mh, incl, vs, per_time_grids=use_per_time_grids)
            # Integrate over frequency (same as model_X does)
            dm = np.dot(sed_band[:-1].T, (vs[1:] - vs[:-1])/vs[:-1])

            # Finish calculating the luminosities from the models:
            L_band = dm 

            # Calculate and add to the likelihood:
            # OPTIMIZATION: Band weighting option - normalize by number of points when enabled
            # When weight_by_band=True: Each band's average chi-squared per point contributes equally
            # This prevents well-sampled bands from dominating the fit
            # When weight_by_band=False: Standard maximum likelihood (each data point weighted equally)
            diff = lum_band - L_band
            var_band = err_band**2.0 + (s * lum_band)**2.0
            band_contribution = -0.5 * ( (diff**2)/var_band  ).sum()
            if self.weight_by_band:
                n_points = len(diff)
                if n_points > 0:
                    band_contribution = band_contribution / n_points
            # Apply per-band weight (default 1.0 if not specified)
            # Weights are applied after normalization (if weight_by_band=True)
            band_weight = self.band_weights.get(band, 1.0) if self.band_weights else 1.0
            likelihood += band_contribution * band_weight

        for band in self.data.bands_X_upperlim:    
            t_band, lum_band, N_sig = self.data.args_band[band]
            v_band = self.data.bands_freq[band]
            
            # OPTIMIZATION: Use pre-computed temperature instead of recalculating
            # Find indices in all_times that correspond to this band's times
            band_indices = find_time_indices(t_band, all_times)
            
            if use_per_time_grids:
                # Extract per-time grids and temperatures for this band
                r_band = [r[i] for i in band_indices]
                T_band = [T_all[i] for i in band_indices]
            else:
                # Extract temperature slice from 2D array
                T_band = T_all[band_indices, :]
                r_band = r
            
            # Compute X-ray model using pre-computed temperature
            vs = np.geomspace(v_band[0]*keV_to_Hz, v_band[1]*keV_to_Hz, num=20)
            sed_band = self.model_SEDs_from_T(t_band, r_band, T_band, log_mh, incl, vs, per_time_grids=use_per_time_grids)
            # Integrate over frequency (same as model_X does)
            dm = np.dot(sed_band[:-1].T, (vs[1:] - vs[:-1])/vs[:-1])

            # Finish calculating the luminosities from the models:
            L_band = dm 

            # Calculate and add to the likelihood:
            # OPTIMIZATION: Band weighting option - normalize by number of points when enabled
            # When weight_by_band=True: Each band's average chi-squared per point contributes equally
            # This prevents well-sampled bands from dominating the fit
            # When weight_by_band=False: Standard maximum likelihood (each data point weighted equally)
            band_contribution = self._log_upper_limit_likelihood(L_band, lum_band, N_sig)
            if self.weight_by_band:
                n_points = len(L_band)
                if n_points > 0:
                    band_contribution = band_contribution / n_points
            # Apply per-band weight (default 1.0 if not specified)
            # Weights are applied after normalization (if weight_by_band=True)
            band_weight = self.band_weights.get(band, 1.0) if self.band_weights else 1.0
            likelihood += band_contribution * band_weight        

        if not np.isnan(likelihood):
            return likelihood
        return -np.inf


