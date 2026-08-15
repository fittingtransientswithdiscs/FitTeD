import os 
import numpy as np
import matplotlib.pyplot as plt 
import pickle 
from collections import OrderedDict
from warnings import warn
from astropy import units
from astropy import cosmology
from .constants import *

__all__ = ["Data_Set"]

manyTDE_available = "No"
usemanyTDE = False
try:
    import manyTDE
except ImportError  as e:
    print(e)
    warn("manyTDE is not avaliable, please install it.  (https://github.com/sjoertvv/manyTDE)", stacklevel=5)
else:
    manyTDE_available = "Yes"
    usemanyTDE = True
    import importlib_resources
    import json
    manyTDEpath = importlib_resources.files("manyTDE") / "data/sources"


class Data_Set:
    def __init__(self, manyTDE_name=None, manyTDE_bands=None, 
                 args_UV=[], bands_UV=[], 
                 args_X=[], bands_X=[], 
                 args_UV_upperlim=[], bands_UV_upperlim=[],
                 args_X_upperlim=[], bands_X_upperlim=[],
                 global_systematic=None,
                 # New parameters for observer-frame flux/magnitude inputs (UV/optical only)
                 args_UV_flux=None, bands_UV_flux=None,
                 args_UV_mag=None, bands_UV_mag=None,
                 args_UV_upperlim_flux=None, bands_UV_upperlim_flux=None,
                 args_UV_upperlim_mag=None, bands_UV_upperlim_mag=None,
                 redshift=None,
                 # New parameters for observer-frame X-ray flux inputs
                 args_X_flux=None, bands_X_flux=None,
                 args_X_upperlim_flux=None, bands_X_upperlim_flux=None
                 ):
        """
            The Data_Set class. This class holds all of the data for a given source. 
            We split the data in "X-ray" data (integrated luminosity across a broad band), 
            and "UV" data (spectral vLv luminosity at a given frequency). 

            Upper limits for both "x-ray" and "UV" data can be included. 

            Inputs:
                manyTDE_name -- IAU name of a TDE. If in manyTDE data set (https://github.com/sjoertvv/manyTDE) then will auto-load data. 
                manyTDE_bands -- which optical/UV bands to use from manyTDE. 

                args_UV -- UV data of the form [times, luminosities, uncertainties, frequencies]. 
                          Times, luminosities, and frequencies should be in rest frame.
                bands_UV -- list of names of each UV band. Needs len(bands_UV) = number of data sets in args_UV. 
                
                args_X -- X-ray data of the form [times, luminosities, uncertainties, [E_low, E_high]]. 
                         Times and luminosities should be in rest frame.
                bands_X -- list of names of each X-ray band. Needs len(bands_X) = number of data sets in args_X. 
        
                args_UV_upperlim -- UV data of the form [times, luminosities, N_sigma, frequencies]. 
                                   Times, luminosities, and frequencies should be in rest frame.
                bands_UV_upperlim -- list of names of each UV band with upperlimits. Needs len(bands_UV_upperlim) = number of data sets in args_UV_upperlim. 

                args_X_upperlim -- X-ray data of the form [times, luminosities, N_sigma, [E_low, E_high]]. 
                                  Times and luminosities should be in rest frame.
                bands_X_upperlim -- list of names of each X-ray band with upperlimits. Needs len(bands_X_upperlim) = number of data sets in args_X_upperlim. 

                global_systematic -- adds an additional factor of (global_systematic * luminosity)^2 to each variance of the data.
                
            New inputs for observer-frame data (UV/optical only):
                args_UV_flux -- Observer-frame fluxes for UV/optical data.
                               Format: List of [times, fluxes_erg/s/cm²/Hz, uncertainties, frequencies_Hz] arrays.
                               All quantities in observer frame. Will be converted to rest-frame vL_v internally.
                bands_UV_flux -- REQUIRED if args_UV_flux provided. List of band names (strings).
                                Must have len(bands_UV_flux) == len(args_UV_flux).
                
                args_UV_mag -- AB magnitudes for UV/optical data.
                              Format: List of [times, AB_magnitudes, uncertainties, frequencies_Hz] arrays.
                              Times and frequencies in observer frame. Will be converted to rest-frame vL_v internally.
                bands_UV_mag -- REQUIRED if args_UV_mag provided. List of band names (strings).
                               Must have len(bands_UV_mag) == len(args_UV_mag).
                
                args_UV_upperlim_flux -- Observer-frame fluxes for UV/optical upper limits.
                                        Format: List of [times, fluxes_erg/s/cm²/Hz, N_sigma, frequencies_Hz] arrays.
                bands_UV_upperlim_flux -- REQUIRED if args_UV_upperlim_flux provided. List of band names (strings).
                
                args_UV_upperlim_mag -- AB magnitudes for UV/optical upper limits.
                                       Format: List of [times, AB_magnitudes, N_sigma, frequencies_Hz] arrays.
                bands_UV_upperlim_mag -- REQUIRED if args_UV_upperlim_mag provided. List of band names (strings).
                
                redshift -- Redshift of the source (required when using flux/mag inputs).
                          If provided, will be stored as self.redshift.
                          Distance will be automatically calculated from redshift using astropy.cosmology.Planck18.
                
            New inputs for observer-frame X-ray data:
                args_X_flux -- Observer-frame integrated X-ray fluxes.
                              Format: List of [times, fluxes_erg/s/cm², uncertainties, [E_low, E_high_keV]] arrays.
                              All quantities in observer frame. Will be converted to rest-frame integrated luminosity internally.
                              WARNING: This uses simplified conversion L = 4π d_L² × F. For accurate analysis,
                              proper K-corrections should be applied.
                bands_X_flux -- REQUIRED if args_X_flux provided. List of band names (strings).
                               Must have len(bands_X_flux) == len(args_X_flux).
                
                args_X_upperlim_flux -- Observer-frame integrated X-ray fluxes for upper limits.
                                      Format: List of [times, fluxes_erg/s/cm², N_sigma, [E_low, E_high_keV]] arrays.
                bands_X_upperlim_flux -- REQUIRED if args_X_upperlim_flux provided. List of band names (strings).
            
            Examples:
                # Using observer-frame fluxes:
                data = Data_Set(
                    args_UV_flux=[[times, fluxes, errors, frequencies]],
                    bands_UV_flux=['g.custom'],
                    redshift=0.1  # Distance calculated automatically from redshift
                )
                
                # Using AB magnitudes:
                data = Data_Set(
                    args_UV_mag=[[times, magnitudes, errors, frequencies]],
                    bands_UV_mag=['r.custom'],
                    redshift=0.1  # Distance calculated automatically from redshift
                )
                
                # Mixing manyTDE with custom flux data:
                data = Data_Set(
                    manyTDE_name='AT2019dsg',
                    manyTDE_bands=['g.ztf', 'r.ztf'],
                    args_UV_flux=[[times, fluxes, errors, frequencies]],
                    bands_UV_flux=['i.custom'],
                    redshift=0.1  # Distance calculated automatically from redshift
                )
                
                # Using observer-frame X-ray flux:
                data = Data_Set(
                    args_X_flux=[[times, xray_fluxes, xray_errors, [0.3, 10.0]]],
                    bands_X_flux=['Swift XRT'],
                    redshift=0.1  # Distance calculated automatically from redshift
                    # Note: A warning will be raised about simplified X-ray conversion
                )
        """

        __lc_color_dict = {}
        for surv in ['ztf','ps','sdss']:
            __lc_color_dict['g.'+surv] = 'g'
            __lc_color_dict['r.'+surv] = 'r'
            __lc_color_dict['i.'+surv] = 'rosybrown'

        __lc_color_dict['W1.wise'] = 'goldenrod'
        __lc_color_dict['W2.wise'] ='saddlebrown'

        __lc_color_dict['UVW2.uvot'] = 'violet'
        __lc_color_dict['UVM2.uvot'] = 'magenta'
        __lc_color_dict['UVW1.uvot'] = 'fuchsia'

        __lc_color_dict['U.uvot'] = 'darkblue'
        __lc_color_dict['u.sdss'] = 'darkblue'

        __lc_color_dict['F125LP'] = 'darkviolet'
        __lc_color_dict['F150LP'] = 'darkviolet'
        __lc_color_dict['F225W'] = 'magenta'

        __lc_color_dict['FUV'] = 'darkviolet'
        __lc_color_dict['NUV'] = 'magenta'

        __lc_color_dict['B.uvot'] = 'lightblue'
        __lc_color_dict['V.uvot'] = 'orange'
        __lc_color_dict['c.atlas'] = 'cyan'
        __lc_color_dict['o.atlas'] = 'orange'

        __all_markers = ['s', 'D',  'p', 'H',  'X', 'P',  '*', 'o']
        __all_upperlim_markers = ['v', '^', '<', '>']
        __back_up_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


        self.args_X=args_X

        if args_X != [] and bands_X == []:
            bands_X = ['X-ray']

        if args_X_upperlim != [] and bands_X_upperlim == []:
            bands_X_upperlim = ['X-ray upper limit']


        self.args_band=OrderedDict([])
        self.bands_freq=OrderedDict([])
        self.bands_systematic=OrderedDict([])
        self.band_colours=OrderedDict([])
        self.band_markers=OrderedDict([])

        ########################################################
        #  Load data directly from manyTDE, and save in args_UV. 
        ########################################################
        if manyTDE_name is not None and usemanyTDE:
            lc_dict, filters, frequencies_Hz, mjd0, z = self.get_lightcurve_data(tde_name=manyTDE_name)
            times, vLvs, errs = [[] for _ in range(len(manyTDE_bands))], [[] for _ in range(len(manyTDE_bands))], [[] for _ in range(len(manyTDE_bands))]
            freqs = np.zeros(len(manyTDE_bands))
            self.redshift = z
            self.mjd0 = mjd0
            self.d_Mpc = self.get_lum_distance(z)

            for i, band in enumerate(manyTDE_bands):
                log = [_band == band for _band in filters]
                if sum(log)>0.5:
                    i_band = np.argmax(log)
                    times[i] = lc_dict[band][0]
                    vLvs[i] = lc_dict[band][1] 
                    errs[i] = lc_dict[band][2] 
                    freqs[i] = frequencies_Hz[i_band] 

                    if args_UV == []: # This will add additional data to args_UV from manyTDE, unless no data uploaded, in which case only manyTDE data will be used. 
                        args_UV = [[times[i], vLvs[i], errs[i], freqs[i]]]
                        bands_UV = [band]
                    else:
                        args_UV += [[times[i], vLvs[i], errs[i], freqs[i]]]
                        bands_UV += [band]
                else:
                    print('The band ` {0} ` is not in the manyTDE data set of source {1}. \n Available bands are {2}'.format(band, manyTDE_name, filters))

        ########################################################
        #  Process observer-frame flux/magnitude inputs (UV/optical only)
        ########################################################
        
        # Store redshift and distance if provided
        # Note: manyTDE already sets self.redshift and self.d_Mpc if manyTDE_name is provided
        if redshift is not None:
            # If redshift provided explicitly, use it (may override manyTDE redshift)
            self.redshift = redshift
            self.d_Mpc = self.get_lum_distance(redshift)
        elif (args_UV_flux is not None or args_UV_mag is not None or 
              args_UV_upperlim_flux is not None or args_UV_upperlim_mag is not None):
            # Check if redshift was set by manyTDE
            if not hasattr(self, 'redshift'):
                raise ValueError("redshift must be provided when using observer-frame flux or magnitude inputs")
            # If manyTDE set redshift, calculate distance from it
            if not hasattr(self, 'd_Mpc'):
                self.d_Mpc = self.get_lum_distance(self.redshift)
        
        # Process args_UV_flux
        if args_UV_flux is not None:
            if bands_UV_flux is None:
                raise ValueError("bands_UV_flux must be provided when args_UV_flux is provided")
            if len(bands_UV_flux) != len(args_UV_flux):
                raise ValueError(f"Length of bands_UV_flux ({len(bands_UV_flux)}) must match length of args_UV_flux ({len(args_UV_flux)})")
            # Use redshift from manyTDE if available, otherwise require it
            z_to_use = redshift if redshift is not None else (self.redshift if hasattr(self, 'redshift') else None)
            if z_to_use is None:
                raise ValueError("redshift must be provided when using observer-frame flux inputs (or use manyTDE_name which provides redshift)")
            
            for i, flux_data in enumerate(args_UV_flux):
                if len(flux_data) != 4:
                    raise ValueError(f"args_UV_flux[{i}] must have format [times, fluxes, uncertainties, frequencies]")
                times_obs, flux_obs, err_obs, freq_obs = flux_data
                times_rest, vL_v, err_rest, freq_rest = self._flux_to_luminosity(
                    times_obs, flux_obs, err_obs, freq_obs, z_to_use
                )
                # Append to args_UV and bands_UV
                if args_UV == []:
                    args_UV = [[times_rest, vL_v, err_rest, freq_rest]]
                    bands_UV = [bands_UV_flux[i]]
                else:
                    args_UV += [[times_rest, vL_v, err_rest, freq_rest]]
                    bands_UV += [bands_UV_flux[i]]
        
        # Process args_UV_mag
        if args_UV_mag is not None:
            if bands_UV_mag is None:
                raise ValueError("bands_UV_mag must be provided when args_UV_mag is provided")
            if len(bands_UV_mag) != len(args_UV_mag):
                raise ValueError(f"Length of bands_UV_mag ({len(bands_UV_mag)}) must match length of args_UV_mag ({len(args_UV_mag)})")
            # Use redshift from manyTDE if available, otherwise require it
            z_to_use = redshift if redshift is not None else (self.redshift if hasattr(self, 'redshift') else None)
            if z_to_use is None:
                raise ValueError("redshift must be provided when using observer-frame magnitude inputs (or use manyTDE_name which provides redshift)")
            
            for i, mag_data in enumerate(args_UV_mag):
                if len(mag_data) != 4:
                    raise ValueError(f"args_UV_mag[{i}] must have format [times, magnitudes, uncertainties, frequencies]")
                times_obs, mag_obs, err_obs, freq_obs = mag_data
                times_rest, vL_v, err_rest, freq_rest = self._ab_mag_to_luminosity(
                    times_obs, mag_obs, err_obs, freq_obs, z_to_use
                )
                # Append to args_UV and bands_UV
                if args_UV == []:
                    args_UV = [[times_rest, vL_v, err_rest, freq_rest]]
                    bands_UV = [bands_UV_mag[i]]
                else:
                    args_UV += [[times_rest, vL_v, err_rest, freq_rest]]
                    bands_UV += [bands_UV_mag[i]]
        
        # Process args_UV_upperlim_flux
        if args_UV_upperlim_flux is not None:
            if bands_UV_upperlim_flux is None:
                raise ValueError("bands_UV_upperlim_flux must be provided when args_UV_upperlim_flux is provided")
            if len(bands_UV_upperlim_flux) != len(args_UV_upperlim_flux):
                raise ValueError(f"Length of bands_UV_upperlim_flux ({len(bands_UV_upperlim_flux)}) must match length of args_UV_upperlim_flux ({len(args_UV_upperlim_flux)})")
            # Use redshift from manyTDE if available, otherwise require it
            z_to_use = redshift if redshift is not None else (self.redshift if hasattr(self, 'redshift') else None)
            if z_to_use is None:
                raise ValueError("redshift must be provided when using observer-frame flux inputs (or use manyTDE_name which provides redshift)")
            
            for i, flux_data in enumerate(args_UV_upperlim_flux):
                if len(flux_data) != 4:
                    raise ValueError(f"args_UV_upperlim_flux[{i}] must have format [times, fluxes, N_sigma, frequencies]")
                times_obs, flux_obs, N_sigma, freq_obs = flux_data
                # For upper limits, convert flux to luminosity (N_sigma stays the same)
                times_rest, vL_v, _, freq_rest = self._flux_to_luminosity(
                    times_obs, flux_obs, np.ones_like(flux_obs), freq_obs, z_to_use
                )
                # N_sigma remains unchanged
                if args_UV_upperlim == []:
                    args_UV_upperlim = [[times_rest, vL_v, N_sigma, freq_rest]]
                    bands_UV_upperlim = [bands_UV_upperlim_flux[i]]
                else:
                    args_UV_upperlim += [[times_rest, vL_v, N_sigma, freq_rest]]
                    bands_UV_upperlim += [bands_UV_upperlim_flux[i]]
        
        # Process args_UV_upperlim_mag
        if args_UV_upperlim_mag is not None:
            if bands_UV_upperlim_mag is None:
                raise ValueError("bands_UV_upperlim_mag must be provided when args_UV_upperlim_mag is provided")
            if len(bands_UV_upperlim_mag) != len(args_UV_upperlim_mag):
                raise ValueError(f"Length of bands_UV_upperlim_mag ({len(bands_UV_upperlim_mag)}) must match length of args_UV_upperlim_mag ({len(args_UV_upperlim_mag)})")
            # Use redshift from manyTDE if available, otherwise require it
            z_to_use = redshift if redshift is not None else (self.redshift if hasattr(self, 'redshift') else None)
            if z_to_use is None:
                raise ValueError("redshift must be provided when using observer-frame magnitude inputs (or use manyTDE_name which provides redshift)")
            
            for i, mag_data in enumerate(args_UV_upperlim_mag):
                if len(mag_data) != 4:
                    raise ValueError(f"args_UV_upperlim_mag[{i}] must have format [times, magnitudes, N_sigma, frequencies]")
                times_obs, mag_obs, N_sigma, freq_obs = mag_data
                # For upper limits, convert magnitude to luminosity (N_sigma stays the same)
                times_rest, vL_v, _, freq_rest = self._ab_mag_to_luminosity(
                    times_obs, mag_obs, np.ones_like(mag_obs), freq_obs, z_to_use
                )
                # N_sigma remains unchanged
                if args_UV_upperlim == []:
                    args_UV_upperlim = [[times_rest, vL_v, N_sigma, freq_rest]]
                    bands_UV_upperlim = [bands_UV_upperlim_mag[i]]
                else:
                    args_UV_upperlim += [[times_rest, vL_v, N_sigma, freq_rest]]
                    bands_UV_upperlim += [bands_UV_upperlim_mag[i]]
        
        ########################################################
        #  Process observer-frame X-ray flux inputs
        ########################################################
        
        # Process args_X_flux
        if args_X_flux is not None:
            if bands_X_flux is None:
                raise ValueError("bands_X_flux must be provided when args_X_flux is provided")
            if len(bands_X_flux) != len(args_X_flux):
                raise ValueError(f"Length of bands_X_flux ({len(bands_X_flux)}) must match length of args_X_flux ({len(args_X_flux)})")
            # Use redshift from manyTDE if available, otherwise require it
            z_to_use = redshift if redshift is not None else (self.redshift if hasattr(self, 'redshift') else None)
            if z_to_use is None:
                raise ValueError("redshift must be provided when using observer-frame X-ray flux inputs (or use manyTDE_name which provides redshift)")
            
            for i, flux_data in enumerate(args_X_flux):
                if len(flux_data) != 4:
                    raise ValueError(f"args_X_flux[{i}] must have format [times, fluxes, uncertainties, [E_low, E_high]]")
                times_obs, flux_obs, err_obs, energy_band = flux_data
                times_rest, lum, err_rest, energy_band_rest = self._flux_to_xray_luminosity(
                    times_obs, flux_obs, err_obs, energy_band, z_to_use
                )
                # Append to args_X and bands_X
                if args_X == []:
                    args_X = [[times_rest, lum, err_rest, energy_band_rest]]
                    bands_X = [bands_X_flux[i]]
                else:
                    args_X += [[times_rest, lum, err_rest, energy_band_rest]]
                    bands_X += [bands_X_flux[i]]
        
        # Process args_X_upperlim_flux
        if args_X_upperlim_flux is not None:
            if bands_X_upperlim_flux is None:
                raise ValueError("bands_X_upperlim_flux must be provided when args_X_upperlim_flux is provided")
            if len(bands_X_upperlim_flux) != len(args_X_upperlim_flux):
                raise ValueError(f"Length of bands_X_upperlim_flux ({len(bands_X_upperlim_flux)}) must match length of args_X_upperlim_flux ({len(args_X_upperlim_flux)})")
            # Use redshift from manyTDE if available, otherwise require it
            z_to_use = redshift if redshift is not None else (self.redshift if hasattr(self, 'redshift') else None)
            if z_to_use is None:
                raise ValueError("redshift must be provided when using observer-frame X-ray flux inputs (or use manyTDE_name which provides redshift)")
            
            for i, flux_data in enumerate(args_X_upperlim_flux):
                if len(flux_data) != 4:
                    raise ValueError(f"args_X_upperlim_flux[{i}] must have format [times, fluxes, N_sigma, [E_low, E_high]]")
                times_obs, flux_obs, N_sigma, energy_band = flux_data
                # For upper limits, convert flux to luminosity (N_sigma stays the same)
                times_rest, lum, _, energy_band_rest = self._flux_to_xray_luminosity(
                    times_obs, flux_obs, np.ones_like(flux_obs), energy_band, z_to_use
                )
                # N_sigma remains unchanged
                if args_X_upperlim == []:
                    args_X_upperlim = [[times_rest, lum, N_sigma, energy_band_rest]]
                    bands_X_upperlim = [bands_X_upperlim_flux[i]]
                else:
                    args_X_upperlim += [[times_rest, lum, N_sigma, energy_band_rest]]
                    bands_X_upperlim += [bands_X_upperlim_flux[i]]
        
        # Check for duplicate band names (warn but don't error).
        # NB: manyTDE bands are appended to bands_UV above, so they must NOT be
        # counted again from manyTDE_bands -- doing so made this warning fire on
        # every single manyTDE load, which trained users to ignore it.
        all_band_names = []
        if bands_UV is not None:
            all_band_names.extend(bands_UV)
        if bands_UV_upperlim is not None:
            all_band_names.extend(bands_UV_upperlim)
        if bands_X is not None:
            all_band_names.extend(bands_X)
        if bands_X_upperlim is not None:
            all_band_names.extend(bands_X_upperlim)
        if len(all_band_names) != len(set(all_band_names)):
            duplicates = [name for name in set(all_band_names) if all_band_names.count(name) > 1]
            warn(f"Duplicate band names found: {duplicates}. This may cause issues when accessing bands.", stacklevel=2)

        self.args_UV=args_UV
        self.args_UV_upperlim=args_UV_upperlim
        self.args_X=args_X
        self.args_X_upperlim=args_X_upperlim

        # Setting the dictionary attributes:
        bands=[]
        tmp_bands_X = []
        tmp_bands_X_upperlim = []

        for i, band in enumerate(bands_X):
            if len(self.args_X[i][0])>0.5:
                self.args_band[band]=self.args_X[i][:3]
                self.bands_freq[band]=self.args_X[i][3]
                self.bands_systematic[band] = 0

                self.band_colours[band] = 'k'
                self.band_markers[band] = __all_markers[i]

                bands.append(band)
                tmp_bands_X.append(band)
            else:
                pass
        
        bands_X = tmp_bands_X

        for i, band in enumerate(bands_X_upperlim):
            if len(self.args_X_upperlim[i][0])>0.5:
                self.args_band[band]=self.args_X_upperlim[i][:3]
                self.bands_freq[band]=self.args_X_upperlim[i][3]

                self.band_colours[band] = 'k'
                self.band_markers[band] = __all_upperlim_markers[i]

                bands.append(band)
                tmp_bands_X_upperlim.append(band)
            else:
                pass
        
        bands_X_upperlim = tmp_bands_X_upperlim


        for i, band in enumerate(bands_UV):
            self.args_band[band]=self.args_UV[i][:3]
            self.bands_freq[band]=self.args_UV[i][3]
            self.bands_systematic[band] = 0

            if band in __lc_color_dict.keys():
                self.band_colours[band] = __lc_color_dict[band]
            else:
                print('No default colour for band {0}, colour for this band currently default matplotlib colour {1}.'.format(band, i))
                self.band_colours[band] = __back_up_colors[i]
            
            self.band_markers[band] = 'o'
            bands.append(band)

        for i, band in enumerate(bands_UV_upperlim):
            self.args_band[band]=self.args_UV_upperlim[i][:3]
            self.bands_freq[band]=self.args_UV_upperlim[i][3]
            self.bands_systematic[band] = 0

            if band in __lc_color_dict.keys():
                self.band_colours[band] = __lc_color_dict[band]
            else:
                print('No default colour for band {0}, colour for this band currently default matplotlib colour {1}.'.format(band, i))
                self.band_colours[band] = __back_up_colors[i]
            
            self.band_markers[band] = 'v'
            bands.append(band)


        self.bands_X=bands_X
        self.bands_X_upperlim = bands_X_upperlim
        self.bands_UV=bands_UV
        self.bands_UV_upperlim=bands_UV_upperlim
        self.bands=bands

        if global_systematic is not None:
            self.global_systematic = global_systematic
        else:
            self.global_systematic = 0.0

        return 
    
    ##############################################
    ## Load light curves straight from manyTDE 
    ##############################################
    def get_lightcurve_data(self, tde_name = 'ASASSN-14li'):
        """
        Input: 
            The TDEs name

        Returns:
            1. A dictionary with all of the light curve data, labelled by observing band. 
            2. A list of lightcurve filters with available data. 
            3. The frequency of the band. 
            4. The time of peak (as estimated by Sjoert)
            5. The source redshift

        Note:
            Taken from manyTDE GitHub page and tweaked. 
        """

        fname = '{0}/{1}.json'.format(manyTDEpath, tde_name)
        try:
            tde_data = json.load(open(fname,'r'))# Load data. 
        except Exception as e:
            print(e)
            return None 

        # These conversion are needed because json doesn't store tuples.
        dt = [tuple(x) for x in tde_data['lightcurve']['dtype']]
        lc_obj = [tuple(x) for x in tde_data['lightcurve']['data']] 

        # Make a recarray. 
        lc_rec = np.array(lc_obj, dtype=dt)
        mjd0 = tde_data['peak_mjd']## could use this to do x-ray and uv lightcurves more carefully
        z = tde_data['z']
        DMpc = self.get_lum_distance(z)
        convert_fac = units.Jy.to(units.erg/units.s/units.Hz/units.cm**2.0)#manyTDE stores data in Jy units
        convert_fac *= 4 * np.pi * (DMpc * units.Mpc.to(units.cm))**2.0


        lc_dict = {}
        filters = tde_data['lightcurve']['filters']
        frequency_Hz = tde_data['lightcurve']['frequency_Hz']

        for i, flt in enumerate(filters):
            try:
                idx = lc_rec['filter']==flt

                flux = lc_rec[idx]['flux_Jy']
                flux_corr = flux / tde_data['extinction']['linear_extinction'][flt]# Correct for extinction. 
                lum_corr = flux_corr * convert_fac * frequency_Hz[i]

                lc_dict[flt] = [lc_rec[idx]['mjd']-mjd0, lum_corr, lc_rec[idx]['e_flux_Jy'] * convert_fac * frequency_Hz[i]]
            except:
                pass
        return lc_dict, filters, frequency_Hz, mjd0, z

    def get_lum_distance(self, z):
        """
        Calculate luminosity distance using astropy cosmology.
        
        Uses Planck 2018 cosmology parameters (Planck18) to calculate luminosity distance.
        This replaces the previous manual integration with a standard, well-tested implementation.
        
        Positional Arguments
        ------------------------------------------------------------------------------------------------------------------------
        z (float):
            redshift to calculate the distance of the source
        
        Returns
        ------------------------------------------------------------------------------------------------------------------------
        dL (float):
            distance in Mpc as calculated from the redshift
        """
        if z <= 0:
            return 1.0  # Mpc (same as current behavior)
        
        # Use Planck 2018 cosmology
        cosmo = cosmology.Planck18
        d_L = cosmo.luminosity_distance(z).to(units.Mpc).value
        return d_L

    ######################################
    ## Conversion utilities for observer-frame to rest-frame
    ######################################

    def _ab_mag_to_flux(self, mag, frequency_Hz):
        """
        Convert AB magnitude to observer-frame flux.
        
        Parameters
        ----------
        mag : float or array
            AB magnitude(s)
        frequency_Hz : float
            Observer-frame frequency in Hz
        
        Returns
        -------
        flux : float or array
            Observer-frame flux in erg/s/cm²/Hz
        """
        # AB magnitude definition: m_AB = -2.5 * log10(f_ν) - 48.6
        # Rearranging: f_ν = 10^(-0.4 * (m_AB + 48.6))
        flux = 10**(-0.4 * (mag + 48.6))  # erg/s/cm²/Hz
        return flux

    def _flux_to_luminosity(self, times_obs, flux_obs, err_obs, frequency_obs_Hz, 
                           redshift):
        """
        Convert observer-frame flux to rest-frame vL_v luminosity.
        
        Also converts times and frequencies from observer frame to rest frame.
        Distance is always calculated from redshift using astropy.cosmology.
        
        Parameters
        ----------
        times_obs : array
            Observer-frame times (days)
        flux_obs : array
            Observer-frame flux in erg/s/cm²/Hz
        err_obs : array
            Observer-frame flux uncertainties in erg/s/cm²/Hz
        frequency_obs_Hz : float
            Observer-frame frequency in Hz
        redshift : float
            Redshift of the source
        
        Returns
        -------
        times_rest : array
            Rest-frame times (days)
        vL_v : array
            Rest-frame vL_v luminosities in erg/s
        err_rest : array
            Rest-frame luminosity uncertainties in erg/s
        frequency_rest_Hz : float
            Rest-frame frequency in Hz
        """
        # Calculate distance from redshift
        distance_Mpc = self.get_lum_distance(redshift)
        
        # Convert times: t_rest = t_obs / (1 + z)
        times_rest = times_obs / (1 + redshift)
        
        # Convert frequencies: ν_rest = ν_obs * (1 + z)
        frequency_rest_Hz = frequency_obs_Hz * (1 + redshift)
        
        # Convert flux to vL_v: vL_v = ν_obs × 4π d_L² × f_ν(ν_obs)
        # Note: The (1+z) factors cancel in vL_v calculation
        d_L_cm = distance_Mpc * units.Mpc.to(units.cm)
        convert_fac = 4 * np.pi * d_L_cm**2.0
        vL_v = flux_obs * convert_fac * frequency_obs_Hz  # erg/s
        
        # Convert errors (same scaling as flux)
        err_rest = err_obs * convert_fac * frequency_obs_Hz  # erg/s
        
        return times_rest, vL_v, err_rest, frequency_rest_Hz

    def _ab_mag_to_luminosity(self, times_obs, mag_obs, err_obs, frequency_obs_Hz,
                              redshift):
        """
        Convert AB magnitude to rest-frame vL_v luminosity.
        
        Combines AB magnitude to flux conversion and flux to luminosity conversion.
        Also converts times and frequencies from observer frame to rest frame.
        Distance is always calculated from redshift using astropy.cosmology.
        
        Parameters
        ----------
        times_obs : array
            Observer-frame times (days)
        mag_obs : array
            AB magnitudes
        err_obs : array
            AB magnitude uncertainties
        frequency_obs_Hz : float
            Observer-frame frequency in Hz
        redshift : float
            Redshift of the source
        
        Returns
        -------
        times_rest : array
            Rest-frame times (days)
        vL_v : array
            Rest-frame vL_v luminosities in erg/s
        err_rest : array
            Rest-frame luminosity uncertainties in erg/s
        frequency_rest_Hz : float
            Rest-frame frequency in Hz
        """
        # Convert AB magnitude to flux
        flux_obs = self._ab_mag_to_flux(mag_obs, frequency_obs_Hz)
        
        # Convert magnitude errors to flux errors
        # For small errors: Δf/f ≈ 0.4 * ln(10) * Δm ≈ 0.921 * Δm
        # More precisely: f = 10^(-0.4*(m+48.6)), so df/dm = -0.4*ln(10)*f
        # For error propagation: σ_f = |df/dm| * σ_m = 0.4*ln(10)*f*σ_m
        err_flux = 0.4 * np.log(10) * flux_obs * err_obs
        
        # Convert flux to luminosity
        return self._flux_to_luminosity(times_obs, flux_obs, err_flux, 
                                       frequency_obs_Hz, redshift)

    def _flux_to_xray_luminosity(self, times_obs, flux_obs, err_obs, energy_band, redshift):
        """
        Convert observer-frame X-ray flux to rest-frame integrated luminosity.
        
        WARNING: This is a simplified conversion. For accurate X-ray analysis,
        proper K-corrections and energy band integration should be performed.
        
        Parameters
        ----------
        times_obs : array
            Observer-frame times (days)
        flux_obs : array
            Observer-frame integrated flux (erg/s/cm²) over energy band
        err_obs : array
            Observer-frame flux uncertainties (erg/s/cm²)
        energy_band : list
            [E_low, E_high] in keV (observer frame)
        redshift : float
            Redshift of the source
        
        Returns
        -------
        times_rest : array
            Rest-frame times (days)
        lum : array
            Rest-frame integrated luminosity (erg/s)
        err_rest : array
            Rest-frame luminosity uncertainties (erg/s)
        energy_band_rest : list
            [E_low, E_high] in keV (rest frame)
        """
        # Warn about simplified conversion
        warn("X-ray flux to luminosity conversion uses simplified formula L = 4π d_L² × F. "
             "For accurate analysis, proper K-corrections and energy band integration should be performed. "
             "See X-ray astronomy references for details.", stacklevel=3)
        
        # Convert times: t_rest = t_obs / (1 + z)
        times_rest = times_obs / (1 + redshift)
        
        # Convert energy band: E_rest = E_obs * (1 + z)
        # E = h*nu, so energies blueshift into the rest frame exactly as
        # frequencies do (cf. the UV path above and gr_disc's g = 1/(1+z)).
        energy_band_rest = [energy_band[0] * (1 + redshift),
                            energy_band[1] * (1 + redshift)]
        
        # Calculate distance from redshift
        distance_Mpc = self.get_lum_distance(redshift)
        
        # Convert flux to luminosity: L = 4π d_L² × F
        d_L_cm = distance_Mpc * units.Mpc.to(units.cm)
        convert_fac = 4 * np.pi * d_L_cm**2.0
        lum = flux_obs * convert_fac  # erg/s
        err_rest = err_obs * convert_fac  # erg/s
        
        return times_rest, lum, err_rest, energy_band_rest


    ######################################
    ## Saving and loading 
    ######################################

    def save(self, name):
        """save class as name_DATA.pickle"""
        file = open(name+'_DATA.pickle','wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        return 

    def load(self, name):
        """ loads name_DATA.pickle"""
        ext = ''
        if name[-7:] != '.pickle':
            ext = '.pickle'
            if name[-5:] != "_DATA":
                name+="_DATA"
        try:
            file = open(name+ext,'rb')
        except Exception as e:
            print(e)
            return None
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
        return 


    ######################################
    ## Data Processing
    ######################################

    def rebin_band(self, band, delta_t, t_start=None, t_end=None):
        """
            Rebins the data from a given band. 

            Input
                band -- band name of currently unbinned data.  
                delta_t -- bin size. 
                t_start -- start time of binning in data set. 
                t_end -- end time of binning in data set.

            Returns -- rebins the data in the band and overwrites band in Data_Set. 
        """

        log = [_band == band for _band in self.bands]
        if sum(log)>0.5:
            times, lums, errs = self.args_band[band]
            if t_start is None:
                t_start = min(times)-0.001
            if t_end is None:
                t_end = max(times)+0.001

            ilow = times<t_start
            ihigh = times>t_end 
            tmp_low_t, tmp_low_l, tmp_low_e = times[ilow], lums[ilow], errs[ilow]
            tmp_high_t, tmp_high_l, tmp_high_e = times[ihigh], lums[ihigh], errs[ihigh]
            binned_t, binned_l, binned_e = self.rebin(times, lums, errs, delta_t, t_start, t_end)

            new_times = np.append(tmp_low_t, np.append(binned_t, tmp_high_t))
            new_lums = np.append(tmp_low_l, np.append(binned_l, tmp_high_l))
            new_errs = np.append(tmp_low_e, np.append(binned_e, tmp_high_e))

            self.args_band[band] = new_times, new_lums, new_errs

        else:
            print('The band ` {0} ` is not in the data set of this source. \n Available bands are {1}'.format(band, self.bands))


    def rebin(self, t, x, errx, dt, t_start, t_end):
        """
            Rebins data. 

            Input
                t, x, errx -- unbinned data.  
                dt -- bin size. 
                t_start -- start time of binning in data set. 
                t_end -- end time of binning in data set.

            Returns -- t_new, x_new, errx_new; the rebinned data.
        """
        n = int(np.ceil((t_end - t_start)/dt))
        if n > 0:
            t_new, x_new, errx_new = np.linspace(t_start, t_end, n), np.zeros(n), np.zeros(n)
            i_delete = []

            for i in range(n):
                if i == n-1:
                    i_want = (t>t_new[i])
                else:
                    i_want = (t>t_new[i])*(t<t_new[i+1])
                if sum(i_want) > 0.5:
                    t_new[i] = np.mean(t[i_want])
                    x_new[i] = np.sum(x[i_want]/errx[i_want]**2)/(np.sum(1/errx[i_want]**2))#np.mean(x[i_want])
                    errx_new[i] = np.sqrt(1/(np.sum(1/errx[i_want]**2)))#np.mean(errx[i_want]) 
                else:
                    i_delete += [i]
            
            if len(i_delete)>0.5:
                t_new, x_new, errx_new = np.delete(t_new, i_delete), np.delete(x_new, i_delete), np.delete(errx_new, i_delete)
            
            return t_new, x_new, errx_new
        return t, x, errx


    def remove_before_time(self, band, t_start):
        """
            Deletes data from a band 'band' before time 't_start'. Overwrites band in Data_Set. 
        """
        log = [_band == band for _band in self.bands]
        if sum(log)>0.5:
            times, lums, errs = self.args_band[band]

            ind_cut = np.searchsorted(times, t_start)

            new_times = times[ind_cut:]
            new_lums = lums[ind_cut:]
            new_errs = errs[ind_cut:]

            self.args_band[band] = new_times, new_lums, new_errs
        else:
            print('The band ` {0} ` is not in the data set of this source. \n Available bands are {1}'.format(band, self.bands))
            
        return

    def remove_after_time(self, band, t_end):
        """
            Deletes data from the band 'band' after time 't_end'. Overwrites band in Data_Set. 
        """
        log = [_band == band for _band in self.bands]
        if sum(log)>0.5:
            times, lums, errs = self.args_band[band]

            ind_cut = np.searchsorted(times, t_end)

            new_times = times[:ind_cut]
            new_lums = lums[:ind_cut]
            new_errs = errs[:ind_cut]

            self.args_band[band] = new_times, new_lums, new_errs
        else:
            print('The band ` {0} ` is not in the data set of this source. \n Available bands are {1}'.format(band, self.bands))
            
        return

    def remove_cut(self, band, t_start, t_end):
        """
            Deletes data from the band 'band' after time 't_start' and before 't_end'. Overwrites band in Data_Set. 
        """
        log = [_band == band for _band in self.bands]
        if sum(log)>0.5:
            times, lums, errs = self.args_band[band]
            ind_cut_bottom = np.searchsorted(times, t_start)
            ind_cut_top = np.searchsorted(times, t_end)

            new_times = np.append(times[:ind_cut_bottom], times[ind_cut_top:])
            new_lums = np.append(lums[:ind_cut_bottom], lums[ind_cut_top:])
            new_errs = np.append(errs[:ind_cut_bottom], errs[ind_cut_top:])

            self.args_band[band] = new_times, new_lums, new_errs
        else:
            print('The band ` {0} ` is not in the data set of this source. \n Available bands are {1}'.format(band, self.bands))

        return 

    def get_band(self, band):
        '''
        Returns the data for a given band.

        Positional Arguments
        ------------------------------------------------------------------------------------------------------------------------
        band (str):
            the label corresponding to the deired band
        
        Returns
        ------------------------------------------------------------------------------------------------------------------------
        args_band[band] (tuple of 3 numpy arrays):
            the tuple of arguments from the list corrresponding to the given band (time, luminosity, error)
        '''
        return self.args_band[band]
    
    ######################################
    ## Data Plotting
    ######################################
  
    def plot_band(self, band, fig=None, upperlim=False, 
            yscale='log', xscale='linear', 
            ylabel=r'$L$ [erg/s]', xlabel=r'Time [days]', 
            ylim=None, xlim=None):
        """
            Plots data from a given band. 

            Input (FitTeD):
                band -- band to be plotted.
            
            Input (matplotlib):
                upperlim--boolean. If true then plots as a 'v', if False then plots errorbar.                

                Otherwise all standard stuff:
                    fig=None, 
                    yscale='log', 
                    xscale='linear', 
                    ylabel=r'$L$ [erg/s]', 
                    xlabel=r'Time [days]', 
                    ylim=None, 
                    xlim=None
                
            Returns -- fig; matplotlib figure class.
        """
        
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        else:
            # Check if figure has axes, if not add one
            axes = fig.get_axes()
            if len(axes) == 0:
                ax = fig.add_subplot()
            else:
                ax = axes[0]

        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        
        t, l, e = self.get_band(band)
        

        if xlim is None:
            x_lim_low = min([x_lims[0], min(t)-10])
            x_lim_high = max([x_lims[1], max(t)+10])
            xlim = (x_lim_low, x_lim_high)
        if ylim is None:
            # Fold the existing axis limits in with this band's range, so that
            # plotting several bands onto one figure widens rather than clips.
            #
            # But a *fresh* axes reports (0.0, 1.0), and folding that 0.0 into a
            # log-scaled lower limit makes matplotlib discard the whole ylim with
            # a warning -- which is why the very first band plotted used to come
            # out autoscaled.  Only use the existing limits when they are usable
            # on the scale we are about to set.
            positive = l[l > 0]
            lows = [min(positive) * 0.67] if len(positive) else []
            highs = [max(l) * 1.5] if len(l) else []
            usable = (yscale != 'log')
            if usable or y_lims[0] > 0:
                lows.append(y_lims[0])
            if usable or y_lims[1] > 0:
                highs.append(y_lims[1])
            ylim = (min(lows), max(highs)) if lows and highs else None
        

        if upperlim:
            ax.errorbar(t, l, fmt=self.band_markers[band], label=band, c=self.band_colours[band])
        else:
            ax.errorbar(t, l, e, fmt=self.band_markers[band], label=band, c=self.band_colours[band])

        ax.set(yscale=yscale, 
            xscale=xscale, 
            ylabel=ylabel, 
            xlabel=xlabel, 
            ylim=ylim, 
            xlim=xlim)
        
        ax.legend()

        return fig 
    
    def plot_data(self,  bands=None,#if bands is None does all bands.  
                fig=None,
                yscale='log', xscale='linear', 
                ylabel=r'$L$ [erg/s]', xlabel=r'Time [days]', 
                ylim=None, xlim=None):
        """
            Plots data from a set of bands. 

            Input (FitTeD):
                bands -- bands to be plotted. If bands is None does all bands.  
            
            Input (matplotlib):
                All standard stuff:
                    fig=None, 
                    yscale='log', 
                    xscale='linear', 
                    ylabel=r'$L$ [erg/s]', 
                    xlabel=r'Time [days]', 
                    ylim=None, 
                    xlim=None
                
            Returns -- fig; matplotlib figure class.
        """        
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        plot_kwargs = dict(fig=fig, 
                           yscale=yscale, 
                           xscale=xscale, 
                           ylabel=ylabel, 
                           xlabel=xlabel, 
                           ylim=ylim, 
                           xlim=xlim
                          )

        for band in self.bands_UV:
            if (bands is not None and band in bands):
                self.plot_band(band, **plot_kwargs)
            elif bands is None:
                self.plot_band(band, **plot_kwargs)

        for band in self.bands_X:
            if (bands is not None and band in bands):
                self.plot_band(band, **plot_kwargs)
            elif bands is None:
                self.plot_band(band, **plot_kwargs)
            
        for band in self.bands_X_upperlim:
            if (bands is not None and band in bands):
                self.plot_band(band, upperlim=True, **plot_kwargs)
            elif bands is None:
                self.plot_band(band, upperlim=True, **plot_kwargs)

        return fig

            
