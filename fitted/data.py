import os 
import numpy as np
import matplotlib.pyplot as plt 
import pickle 
from scipy.integrate import simpson
from collections import OrderedDict
from warnings import warn
from astropy import units
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
                 global_systematic=None
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
                bands_UV -- list of names of each UV band. Needs len(bands_UV) = number of data sets in args_UV. 
                
                args_X -- X-ray data of the form [times, luminosities, uncertainties, [E_low, E_high]]. 
                bands_X -- list of names of each X-ray band. Needs len(bands_X) = number of data sets in args_X. 
        
                args_UV_upperlim -- UV data of the form [times, luminosities, N_sigma, frequencies]. 
                bands_UV_upperlim -- list of names of each UV band with upperlimits. Needs len(bands_UV_upperlim) = number of data sets in args_UV_upperlim. 

                args_X_upperlim -- X-ray data of the form [times, luminosities, N_sigma, [E_low, E_high]]. 
                bands_X_upperlim -- list of names of each X-ray band with upperlimits. Needs len(bands_X_upperlim) = number of data sets in args_X_upperlim. 

                global_systematic -- adds an additional factor of (global_systematic * luminosity)^2 to each variance of the data. 
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


        self.args_UV=args_UV
        self.args_UV_upperlim=args_UV_upperlim
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
                print('No default colour for band {1}, colour for this band currently default matplotlib colour {2}.'.format(band, i))
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
        This uses the most recent Planck values of the cosmological parameters and redshift to get distance in Mpc. 
        
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
            dL = 1
        else:            

            dH = c*1e-3/H0 # Mpc

            z_int = np.logspace(-8, np.log10(z), 100000)
            dL = (1 + z) * dH * simpson( 1/(omega_m * (1 + z_int)**3 + omega_l)**0.5, z_int )
        return dL


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
                    x_new[i] = np.mean(x[i_want])
                    errx_new[i] = np.mean(errx[i_want]) 
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
            ax = fig.get_axes()[0]

        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        
        t, l, e = self.get_band(band)
        

        if xlim is None:
            x_lim_low = min([x_lims[0], min(t)-10])
            x_lim_high = max([x_lims[1], max(t)+10])
            xlim = (x_lim_low, x_lim_high)
        if ylim is None:
            y_lim_low = min([y_lims[0], min(l[l>0])*0.67])
            y_lim_high = max([y_lims[1], max(l)*1.5])
            ylim = (y_lim_low, y_lim_high)
        

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

            
