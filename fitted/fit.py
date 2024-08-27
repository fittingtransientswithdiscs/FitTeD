import numpy as np
from scipy.optimize import minimize
import emcee
import pickle
import matplotlib.pyplot as plt 
import corner
from . import prior as pr
from .models import *
import os
from .constants import *

__all__ = ["NoFitYetError", "Fit"]


class NoFitYetError(Exception):
    pass

class Fit():
    def __init__(self, model=None, prior=None):
        """
            FitTeD Fit class. The class which controls statistical analysis of the data, and fitting procedures. 
            Fit class also contains all of the analysis plotting methods. 

            Inputs:
                model -- FitTeD model class. If None, defaults to GR_disc with zero data. 
                prior -- input parameter prior. If none, defaults to window prior.
        """
        if model is None:
            self.model = GR_disc(data=None, 
              colour_correction=True, 
              rest_frame=True, source_redshift=None, 
              decay=True, decay_type='pl', 
              rise=False, rise_type='gauss')
        elif isinstance(model, Model_base):
            self.model = model
        else:
            raise TypeError("Model must be of type " + str(Model_base) )
        
        if prior is None:
            self._log_prior = pr.log_window_prior(    bounds = self.model.default_bounds, 
                                                            key_parameters = self.model.default_key_pars, 
                                                            early_parameters = self.model.default_early_pars)
        else:
            self._log_prior = prior
        
        self._log_likelihood = self.model.log_likelihood
        self.last_best_fit = None
        self.sampler = None
        self.chain = None
        self.chain_probabilities = None
        return 

    ######################################
    ## Saving and loading 
    ######################################

    def save(self, name):
        """save class as name_FIT.pickle"""
        file = open(name+'_FIT.pickle','wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        return 

    def load(self, name):
        """try load name_FIT.pickle"""
        ext = ''
        if name[-7:] != '.pickle':
            ext = '.pickle'
            if name[-4:] != "_FIT":
                name+="_FIT"
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
    ## Plotting data, models and analysis
    ######################################
        
    def format_plots(self, usetex=False):
        """
            Formats all plots as default.  
        """
        self.model.data.format_plots()
        return 
    
    def plot_data(self, fig=None, bands=None,#if bands is None does all bands.  
            yscale='log', xscale='linear', 
            ylabel=r'$L$ [erg/s]', xlabel=r'Time [days]', 
            ylim=None, xlim=None):
        """
            Plots data from a set of given bands. Uses Data_Set class method.

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

        fig = self.model.data.plot_data(fig=fig, bands=bands, 
            yscale=yscale, xscale=xscale, ylabel=ylabel, xlabel=xlabel, ylim=ylim, xlim=xlim)
        
        return fig

    def plot_band(self, band, fig=None, upperlim=False, 
            yscale='log', xscale='linear', 
            ylabel=r'$L$ [erg/s]', xlabel=r'Time [days]', 
            ylim=None, xlim=None):
        """
            Plots data from a  given bands. Uses Data_Set class method.

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

        fig = self.model.data.plot_band(fig=fig, band=band, 
        yscale=yscale, xscale=xscale, ylabel=ylabel, xlabel=xlabel, ylim=ylim, xlim=xlim, upperlim=upperlim)
        
        return fig


    def plot_corner(self, just_disc=True, f_discard=0.5, fig=None, color=None):#... other options. 
        """
            Plots emcee output as a corner plot.

            Input:
                just_disc -- boolean. If true, only does disc parameters. If false, includes non-disc. 
                f_discard -- factor f, 0<f<1, fraction of chain to discard for burn in. 
            
            Input (matplotlib):
                All standard stuff:
                    fig=None, 
                    color=None. 

            Returns -- fig; matplotlib figure class.
        """
        
        samples = self.chain
        ndim = len(samples[0, 0])
        n_discard = int(len(samples) * f_discard)
        flat_samples = samples[n_discard:].reshape((-1, ndim))
        if just_disc:
            labels=[r"$\log M_\bullet$", r"$a_\bullet$", r"$M_{\rm disc}$", r"$r_0$", r"$t_{\rm visc}$", r"$t_0$",r"$i$"]
            i_corner = np.asarray([flat_samples[:, iii] for iii in range(7)]).T
            fig = corner.corner(i_corner, color=color, plot_contours=True, plot_datapoints=False, smooth=True, labels=labels, density=True, fig=fig)
        else:
            if not self.model.rise:
                if self.model.decay_type == 'exp':
                    labels=[r"$\log M_\bullet$", r"$a_\bullet$", r"$M_{\rm disc}$", r"$r_0$", r"$t_{\rm visc}$", r"$t_0$", r"$i$", r"$\log L$", r"$t_{\rm decay}$", r"$\log T$"]
                elif self.model.decay_type == 'pl':
                    labels=[r"$\log M_\bullet$", r"$a_\bullet$", r"$M_{\rm disc}$", r"$r_0$", r"$t_{\rm visc}$", r"$t_0$", r"$i$", r"$\log L$", r"$t_{\rm fb}$", r"$p$", r"$\log T$"]
            else:
                if self.model.decay_type == 'exp':
                    labels=[r"$\log M_\bullet$", r"$a_\bullet$", r"$M_{\rm disc}$", r"$r_0$", r"$t_{\rm visc}$", r"$t_0$", r"$i$", r"$\log L$", r"$t_{\rm decay}$", r"$t_{\rm peak}$", r"$\sigma_{\rm rise}$", r"$\log T$"]
                elif self.model.decay_type == 'pl':
                    labels=[r"$\log M_\bullet$", r"$a_\bullet$", r"$M_{\rm disc}$", r"$r_0$", r"$t_{\rm visc}$", r"$t_0$", r"$i$", r"$\log L$", r"$t_{\rm fb}$", r"$p$", r"$t_{\rm peak}$", r"$\sigma_{\rm rise}$", r"$\log T$"]
            fig = corner.corner(flat_samples, color=color, plot_contours=True, plot_datapoints=False, smooth=True, labels=labels, density=True, fig=fig)
        
        return fig


    def plot_walkers(self, just_disc=True, fig=None, color='k', alpha=0.3):#... other options. 
        """
            Plots emcee walkers.

            Input:
                just_disc -- boolean. If true, only does disc parameters. If false, includes non-disc. 
            
            Input (matplotlib):
                All standard stuff:
                    fig=None, 
                    color='k',
                    alpha=0.3.  

            Returns -- fig; matplotlib figure class.
        """
        
        samples = self.chain
        if just_disc:
            labels=[r"$\log M_\bullet$", r"$a_\bullet$", r"$M_{\rm disc}$", r"$r_0$", r"$t_{\rm visc}$", r"$t_0$",r"$i$"]
        else:
            if not self.model.rise:
                if self.model.decay_type == 'exp':
                    labels=[r"$\log M_\bullet$", r"$a_\bullet$", r"$M_{\rm disc}$", r"$r_0$", r"$t_{\rm visc}$", r"$t_0$", r"$i$", r"$\log L$", r"$t_{\rm decay}$", r"$\log T$"]
                elif self.model.decay_type == 'pl':
                    labels=[r"$\log M_\bullet$", r"$a_\bullet$", r"$M_{\rm disc}$", r"$r_0$", r"$t_{\rm visc}$", r"$t_0$", r"$i$", r"$\log L$", r"$t_{\rm fb}$", r"$p$", r"$\log T$"]
            else:
                if self.model.decay_type == 'exp':
                    labels=[r"$\log M_\bullet$", r"$a_\bullet$", r"$M_{\rm disc}$", r"$r_0$", r"$t_{\rm visc}$", r"$t_0$", r"$i$", r"$\log L$", r"$t_{\rm decay}$", r"$t_{\rm peak}$", r"$\sigma_{\rm rise}$", r"$\log T$"]
                elif self.model.decay_type == 'pl':
                    labels=[r"$\log M_\bullet$", r"$a_\bullet$", r"$M_{\rm disc}$", r"$r_0$", r"$t_{\rm visc}$", r"$t_0$", r"$i$", r"$\log L$", r"$t_{\rm fb}$", r"$p$", r"$t_{\rm peak}$", r"$\sigma_{\rm rise}$", r"$\log T$"]
        
        if fig is None:
            fig, axes = plt.subplots(len(labels), sharex=True, figsize=(3*len(labels), 2*len(labels)))
        else:
            axes = fig.get_axes()

        for i in range(len(labels)):
            ax = axes[i]
            ax.plot(samples[:, :, i], color=color, alpha=alpha)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step number")
        return fig


    def plot_posterior_lightcurves(self, t_plot, bands=None, fig=None, 
                                N=100, f_discard=0.5, plot_median=True, 
                                as_range=True, plotted_range=[18, 84], 
                                ignore_sigma=3, 
                                show_disc=True, show_early=False, show_total=True):
        """
            Plots emcee output as lightcurve posteriors.

            Input:
                t_plot -- times to evaluate light curves at. 
                bands -- bands to plot data and model. 
                f_discard -- factor f, 0<f<1, fraction of chain to discard for burn in. 
                N -- number of posteriors to sample from chain. 
                
                ignore_sigma -- sigma clipping. Do not plot posteriors if deviations of > ignore_sigma from posterior medians of chain. 

                plot_median -- boolean. If true diplays light curve of posterior median. 
                as_range -- boolean. If true plots as shaded posterior. If false plots all posterior lines. 
                plotted_range -- [p_low, p_high], the percentage range to plot as shaded region if as_ramge = True. 

                Choice of which light curve components to plot (all boolean):    
                    show_disc -- default True. 
                    show_early -- default False. 
                    show_total -- default True. 

            Input (matplotlib):
                fig=None, 

            Returns -- fig; matplotlib figure class.
        """
        
        samples = self.chain 
        ndim = len(samples[0, 0])
        n_discard = int(len(samples) * f_discard)
        flat_samples = samples[n_discard:].reshape((-1, ndim))
        medians = np.median(flat_samples, axis=0)
        stds = np.std(flat_samples, axis=0)

        if bands is not None:
            lms = np.zeros(len(t_plot)*N*len(bands)).reshape(len(t_plot), N, len(bands))
            les = np.zeros(len(t_plot)*N*len(bands)).reshape(len(t_plot), N, len(bands))
        else:
            bands = self.model.data.bands
            lms = np.zeros(len(t_plot)*N*len(bands)).reshape(len(t_plot), N, len(bands))
            les = np.zeros(len(t_plot)*N*len(bands)).reshape(len(t_plot), N, len(bands))

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        else:
            ax = fig.get_axes()[0]

        m = self.model
        d = m.data
        k = 0
        _done = 0

        while k < N:
            j = np.random.randint(len(flat_samples))
            if not m.rise:
                if m.decay:
                    if m.decay_type == 'exp':
                        log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, log_T = flat_samples[-j]
                    elif m.decay_type == 'pl':
                        log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, p, log_T = flat_samples[-j]
                else:
                    log_mh, a_bh, m_disc, r0, tvi, t0, incl = flat_samples[-j]
            else:
                if m.decay:
                    if m.decay_type == 'exp':
                        log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, t_peak, sigma, log_T = flat_samples[-j]
                    elif m.decay_type == 'pl':
                        log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, p, t_peak, sigma, log_T = flat_samples[-j]
                else:
                    log_mh, a_bh, m_disc, r0, tvi, t0, incl = flat_samples[-j]
                    


            delta =  abs((flat_samples[-j]-medians)/stds)
            if (delta>ignore_sigma).any():
                pass
            else:
                for ii, band in enumerate(bands):
                    if band in self.model.data.bands_UV:
                        lmod = m.model_UV(t_plot, log_mh, a_bh, m_disc, r0, tvi, t0, incl, v=d.bands_freq[band])
                        if not m.rise:
                            if m.decay:
                                if m.decay_type == 'exp':
                                    emod = m.decay_model(t_plot, log_L, tdecay, log_T, v=d.bands_freq[band])
                                elif m.decay_type == 'pl':
                                    emod = m.decay_model(t_plot, log_L, tdecay, p, log_T, v=d.bands_freq[band])
                            else:
                                emod = m.decay_model(t_plot)
                        else:
                            if m.decay:
                                if m.decay_type == 'exp':
                                    emod = m.decay_model(t_plot, log_L, tdecay, t_peak, log_T, v=d.bands_freq[band]) + m.rise_model(t_plot, log_L, sigma, t_peak, log_T, v=d.bands_freq[band])
                                elif m.decay_type == 'pl':
                                    emod = m.decay_model(t_plot, log_L, tdecay, p, t_peak, log_T, v=d.bands_freq[band]) + m.rise_model(t_plot, log_L, sigma, t_peak, log_T, v=d.bands_freq[band])
                            else:
                                emod = m.decay_model(t_plot)
                        if np.isnan(lmod).any():
                            pass
                        else:
                            lms[:, k, ii] = lmod
                            les[:, k, ii] = emod
                            _done += 1


                    else:
                        lmod = m.model_X(t_plot, log_mh, a_bh, m_disc, r0, tvi, t0, incl, El=d.bands_freq[band][0], Eh=d.bands_freq[band][1])
                        if np.isnan(lmod).any():
                            pass
                        else:
                            lms[:, k, ii] = lmod
                            _done += 1

                if _done == len(bands):
                    k+=1
                    _done = 0
        
        if not as_range:
            for k in range(N):
                for ii, band in enumerate(bands):
                    c = d.band_colours[band]
                    if show_disc:
                        ax.plot(t_plot, lms[:, k, ii], '--', c=c, alpha=0.1)
                    if show_total:
                        ax.plot(t_plot, lms[:, k, ii]+les[:, k, ii], '-', c=c, alpha=0.1)
                    if show_early:
                        ax.plot(t_plot, les[:, k, ii], '-.', c=c, alpha=0.1)

        else:
            for ii, band in enumerate(bands):
                c = d.band_colours[band]
                post_m = np.percentile(lms[:, :, ii], plotted_range, axis=1)
                post_e = np.percentile(les[:, :, ii], plotted_range, axis=1)
                if show_disc:            
                    ax.fill_between(t_plot, post_m[0], post_m[1], color=c, alpha=0.1)
                if show_total:
                    ax.fill_between(t_plot, post_m[0]+post_e[0], post_m[1]+post_e[1], color=c, alpha=0.1)
                if show_early:
                    ax.fill_between(t_plot, post_e[0], post_e[1], color=c, alpha=0.1)


        if plot_median:
            for ii, band in enumerate(bands):
                c = d.band_colours[band]
                post_m = np.percentile(lms[:, :, ii], [50], axis=1)
                post_e = np.percentile(les[:, :, ii], [50], axis=1)
                if show_disc:
                    ax.plot(t_plot, post_m[0], color=c, alpha=1, ls='--')
                if show_total:
                    ax.plot(t_plot, post_m[0]+post_e[0], color=c, alpha=1)
                if show_early:
                    ax.plot(t_plot, post_e[0], color=c, alpha=1, ls='-.')


        return fig


    def plot_bolometric_luminosity_posterior(self, t_plot, fig=None, as_eddington=False, 
                                N=100, f_discard=0.5, plot_median=True, 
                                as_range=True, plotted_range=[18, 84], 
                                ignore_sigma=3, yscale='log', xscale='log', 
                                ylabel=r'$L_{\rm bol}$ [erg/s]', xlabel=r'Time [days]', 
                                ylim=None, xlim=None, color='k', alpha=0.3):
        """
            Plots emcee output as bolometric luminosity posteriors. For analysis purposes.

            Input:
                t_plot -- times to evaluate luminosity at. 
                f_discard -- factor f, 0<f<1, fraction of chain to discard for burn in. 
                N -- number of posteriors to sample from chain. 
                as_eddington -- boolean, if True plots as Eddington ratio. 
                
                ignore_sigma -- sigma clipping. Do not plot posteriors if deviations of > ignore_sigma from posterior medians of chain. 

                plot_median -- boolean. If true diplays light curve of posterior median. 
                as_range -- boolean. If true plots as shaded posterior. If false plots all posterior lines. 
                plotted_range -- [p_low, p_high], the percentage range to plot as shaded region if as_ramge = True. 

                Choice of which light curve components to plot (all boolean):    
                    show_disc -- default True. 
                    show_early -- default False. 
                    show_total -- default True. 

            Input (matplotlib):
                fig=None, (if None makes a figure, if given plots on figure). 
                yscale='log', 
                xscale='log', 
                ylabel=r'$L_{\rm bol}$ [erg/s]', Will change automatically if as_eddington=True.
                xlabel=r'Time [days]', 
                ylim=None, 
                xlim=None, 
                color='k', 
                alpha=0.3 (of shaded region). 

            Returns -- fig; matplotlib figure class.
        """
                
        samples = self.chain
        ndim = len(samples[0, 0])
        n_discard = int(len(samples) * f_discard)
        flat_samples = samples[n_discard:].reshape((-1, ndim))
        medians = np.median(flat_samples, axis=0)
        stds = np.std(flat_samples, axis=0)
        m = self.model 


        ls = np.zeros(N * len(t_plot)).reshape(N, len(t_plot))
        k=0
        while k < N:
            j = np.random.randint(len(flat_samples))
            delta =  abs((flat_samples[-j][:6]-medians[:6])/stds[:6])
            if (delta>ignore_sigma).any():
                pass
            else:
                log_mh, a_bh, m_disc, r0, tvi, t0 = flat_samples[-j][:6]
                if as_eddington:
                    ls[k, :] = m.get_EddingtonRatio(t_plot-t0, log_mh, a_bh, m_disc, r0, tvi, t0)
                else:
                    ls[k, :] = m.get_Bolometric(t_plot-t0, log_mh, a_bh, m_disc, r0, tvi, t0)
                k+=1
        
        ds = np.percentile(ls, plotted_range, axis=0)
        med = np.percentile(ls, [50], axis=0)

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        else:
            ax = fig.get_axes()[0]
        
        if as_range:
            ax.fill_between(t_plot, ds[0], ds[1], color=color, alpha=alpha)
        else:
            for j in range(k):
                ax.plot(t_plot, ls[k, :], c=color, alpha=alpha)

        if plot_median:
            ax.plot(t_plot, med[0], c=color)


        if as_eddington and (ylabel==r'$L_{\rm bol}$ [erg/s]'):
            ylabel=r'$L_{\rm bol}/L_{\rm edd}$'

        ax.set(yscale=yscale, 
        xscale=xscale, 
        ylabel=ylabel, 
        xlabel=xlabel, 
        ylim=ylim, 
        xlim=xlim)
    
        return fig 


    def plot_isco_accretion_rate_posterior(self, t_plot, fig=None, as_eddington=True, 
                                N=100, f_discard=0.5, plot_median=True, 
                                as_range=True, plotted_range=[18, 84], 
                                ignore_sigma=3, yscale='log', xscale='log', 
                                ylabel=r'$\dot M_{\rm acc}(r_I)$ [g/s]', xlabel=r'Time [days]', 
                                ylim=None, xlim=None, color='k', alpha=0.3):
        """
            Plots emcee output as isco accretion rate posteriors. For analysis purposes.

            Input:
                t_plot -- times to evaluate accretion rate at. 
                f_discard -- factor f, 0<f<1, fraction of chain to discard for burn in. 
                N -- number of posteriors to sample from chain. 
                as_eddington -- boolean, if True plots as Eddington ratio. 
                
                ignore_sigma -- sigma clipping. Do not plot posteriors if deviations of > ignore_sigma from posterior medians of chain. 

                plot_median -- boolean. If true diplays light curve of posterior median. 
                as_range -- boolean. If true plots as shaded posterior. If false plots all posterior lines. 
                plotted_range -- [p_low, p_high], the percentage range to plot as shaded region if as_ramge = True. 

                Choice of which light curve components to plot (all boolean):    
                    show_disc -- default True. 
                    show_early -- default False. 
                    show_total -- default True. 

            Input (matplotlib):
                fig=None, (if None makes a figure, if given plots on figure). 
                yscale='log', 
                xscale='log', 
                ylabel=r'\dot M_{\rm acc}(r_I)$ [g/s]', Will change automatically if as_eddington=True.
                xlabel=r'Time [days]', 
                ylim=None, 
                xlim=None, 
                color='k', 
                alpha=0.3 (of shaded region). 

            Returns -- fig; matplotlib figure class.
        """
        
        samples = self.chain
        ndim = len(samples[0, 0])
        n_discard = int(len(samples) * f_discard)
        flat_samples = samples[n_discard:].reshape((-1, ndim))
        medians = np.median(flat_samples, axis=0)
        stds = np.std(flat_samples, axis=0)
        m = self.model 


        mdots = np.zeros(N * len(t_plot)).reshape(N, len(t_plot))
        k=0
        while k < N:
            j = np.random.randint(len(flat_samples))
            delta =  abs((flat_samples[-j][:6]-medians[:6])/stds[:6])
            if (delta>ignore_sigma).any():
                pass
            else:
                log_mh, a_bh, m_disc, r0, tvi, t0 = flat_samples[-j][:6]
                if as_eddington:
                    mdots[k, :] = -m.get_EddingtonAccretionRatio(t_plot-t0, log_mh, a_bh, m_disc, r0, tvi, t0)[1][:, 1]## negative sign as accretion inwards
                else:
                    mdots[k, :] = -m.get_Mdot(t_plot-t0, log_mh, a_bh, m_disc, r0, tvi, t0)[1][:, 1] * 1e3## negative sign as accretion inwards, 1e3 to cgs. 
                k+=1
        
        ds = np.percentile(mdots, plotted_range, axis=0)
        med = np.percentile(mdots, [50], axis=0)

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        else:
            ax = fig.get_axes()[0]
        
        if as_range:
            ax.fill_between(t_plot, ds[0], ds[1], color=color, alpha=alpha)
        else:
            for j in range(k):
                ax.plot(t_plot, mdots[k, :], c=color, alpha=alpha)

        if plot_median:
            ax.plot(t_plot, med[0], c=color)


        if as_eddington and (ylabel==r'$\dot M_{\rm acc}(r_I)$ [g/s]'):
            ylabel=r'$\dot M_{\rm acc}(r_I)/\dot M_{\rm edd}$'

        ax.set(yscale=yscale, 
        xscale=xscale, 
        ylabel=ylabel, 
        xlabel=xlabel, 
        ylim=ylim, 
        xlim=xlim)
    
        return fig 


    def plot_density_posterior(self, t_plot, fig=None, in_cgs=False, 
                                N=100, f_discard=0.5, plot_median=True, 
                                as_range=True, plotted_range=[18, 84], 
                                ignore_sigma=3, yscale='log', xscale='log', 
                                ylabel=r'$\Sigma$ [g/cm$^2$]', xlabel=r'$r/r_g$', 
                                ylim=None, xlim=None, colors=['b'], alpha=0.3, 
                                rmin=6, rmax=2000, legend=True):
        """
            Plots emcee output as disc density posteriors. For analysis purposes.

            Input:
                t_plot -- times to evaluate density at. 
                f_discard -- factor f, 0<f<1, fraction of chain to discard for burn in. 
                N -- number of posteriors to sample from chain. 
                in_cgs -- boolean, if True plots as radius in cm. 

                rmin = 6 -- inner disc radius to try and plot to.
                rmax = 2000 -- outer disc radius to try and plot to.
                
                ignore_sigma -- sigma clipping. Do not plot posteriors if deviations of > ignore_sigma from posterior medians of chain. 

                plot_median -- boolean. If true diplays light curve of posterior median. 
                as_range -- boolean. If true plots as shaded posterior. If false plots all posterior lines. 
                plotted_range -- [p_low, p_high], the percentage range to plot as shaded region if as_ramge = True. 

                Choice of which light curve components to plot (all boolean):    
                    show_disc -- default True. 
                    show_early -- default False. 
                    show_total -- default True. 

            Input (matplotlib):
                fig=None, (if None makes a figure, if given plots on figure). 
                yscale='log', 
                xscale='log', 
                ylabel=r'$\Sigma$ [g/cm$^2$]', 
                xlabel=r'$r/r_g$', Will change automatically if as_eddington=True.
                ylim=None, 
                xlim=None, 
                color=['b'] must be list as long as t_plot. 
                alpha=0.3 (of shaded region),
                legend=True (show legend). 


            Returns -- fig; matplotlib figure class.
        """
        
        samples = self.chain
        ndim = len(samples[0, 0])
        n_discard = int(len(samples) * f_discard)
        flat_samples = samples[n_discard:].reshape((-1, ndim))
        medians = np.median(flat_samples, axis=0)
        stds = np.std(flat_samples, axis=0)
        m = self.model 

        Nr = 10000
        Nrp = 400

        if type(t_plot) != type([]):
            if type(t_plot) != type(np.asarray([])):
                t_plot = [t_plot]

        ss = np.zeros(len(t_plot)*N*Nr).reshape(len(t_plot), Nr, N)
        rr = np.zeros(N*Nr).reshape(Nr, N)
        rr2 = np.zeros(N*Nr).reshape(Nr, N)

        k=0

        while k < N:
            j = np.random.randint(len(flat_samples))
            delta =  abs((flat_samples[-j][:6]-medians[:6])/stds[:6])
            if (delta>ignore_sigma).any():
                pass
            else:
                log_mh, a_bh, m_disc, r0, tvi, t0 = flat_samples[-j][:6]
                r, s = m.get_Density(t_plot-t0, log_mh, a_bh, m_disc, r0, tvi, t0, N=Nr)
                rr[:, k] = r
                rr2[:, k] = r * G * 10**log_mh * Ms /c**2.0 * 100## cgs units
                for i in range(len(t_plot)):
                    ss[i, :, k] = s[i] * 0.1## cgs units

                k+=1


        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        else:
            ax = fig.get_axes()[0]
        

        
        r_plot = np.geomspace(rmin, rmax, Nrp)
        r_plot2 = r_plot * G * 10**medians[0] * Ms /c**2.0 * 100

        s_plot_up = np.zeros(len(t_plot) * len(r_plot)).reshape(len(t_plot), len(r_plot))
        s_plot_down = np.zeros(len(t_plot) * len(r_plot)).reshape(len(t_plot), len(r_plot))
        s_plot_mid = np.zeros(len(t_plot) * len(r_plot)).reshape(len(t_plot), len(r_plot))

        s_plot_up2 = np.zeros(len(t_plot) * len(r_plot2)).reshape(len(t_plot), len(r_plot2))
        s_plot_down2 = np.zeros(len(t_plot) * len(r_plot2)).reshape(len(t_plot), len(r_plot2))
        s_plot_mid2 = np.zeros(len(t_plot) * len(r_plot2)).reshape(len(t_plot), len(r_plot2))

        for i, r in enumerate(r_plot):
            i_list = np.argmin(abs(rr-r), axis=0)
            
            for j in range(len(t_plot)):
                sr = ss[j, i_list, np.arange(N)]
                ds = np.percentile(sr, plotted_range)
                meds = np.percentile(sr, [50])

                s_plot_up[j, i] = ds[1]
                s_plot_down[j, i] = ds[0]
                s_plot_mid[j, i] = meds[0]

        for i, r in enumerate(r_plot2):
            i_list = np.argmin(abs(rr2-r), axis=0)
            
            for j in range(len(t_plot)):
                sr = ss[j, i_list, np.arange(N)]
                ds = np.percentile(sr, plotted_range)
                meds = np.percentile(sr, [50])

                s_plot_up2[j, i] = ds[1]
                s_plot_down2[j, i] = ds[0]
                s_plot_mid2[j, i] = meds[0]

            
        if as_range:
            if not in_cgs:
                for i in range(len(t_plot)):
                    ax.fill_between(r_plot, s_plot_down[i, :], s_plot_up[i, :], color=colors[i], alpha=alpha)
                    if plot_median:
                        ax.plot(r_plot, s_plot_mid[i, :], color=colors[i], label=r'$t-t_{\rm peak} = %.0d$ [days]'%t_plot[i])
            else:
                for i in range(len(t_plot)):
                    ax.fill_between(r_plot2, s_plot_down2[i, :], s_plot_up2[i, :], color=colors[i], alpha=alpha)
                    if plot_median:
                        ax.plot(r_plot2, s_plot_mid2[i, :], color=colors[i], label=r'$t-t_{\rm peak} = %.0d$ [days]'%t_plot[i])
        else:
            if not in_cgs:
                for i in range(len(t_plot)):
                    for k in range(N):
                        ax.plot(rr[:, k], ss[i, :, k], color=colors[i], alpha=alpha)
                    if plot_median:
                        ax.plot(r_plot, s_plot_mid[i, :], color=colors[i], label=r'$t-t_{\rm peak} = %.0d$ [days]'%t_plot[i])
            else:
                for i in range(len(t_plot)):
                    for k in range(N):
                        ax.plot(rr2[:, k], ss[i, :, k], color=colors[i], alpha=alpha)
                    if plot_median:
                        ax.plot(r_plot2, s_plot_mid2[i, :], color=colors[i], label=r'$t-t_{\rm peak} = %.0d$ [days]'%t_plot[i])


        if in_cgs and (xlabel==r'$r/r_g$'):
            xlabel=r'$r$ [cm]'
        if ylim is None:
            ylim = (1e-5 * max(s_plot_mid[0, :]), 10 * max(s_plot_mid[0, :]))
        ax.set(yscale=yscale, 
        xscale=xscale, 
        ylabel=ylabel, 
        xlabel=xlabel, 
        ylim=ylim, 
        xlim=xlim)

        if legend:
            ax.legend()

        return fig

    ######################################
    ## Log likelihoods of the model
    ######################################
    
    def log_prior(self, pars):
        """
            prior probability ~ p(pars)
        """
        return self._log_prior(pars)
        
    def log_likelihood(self, pars, *args, **kwargs):
        """
            Likelihood ~ p(data | pars)
        """
        return self._log_likelihood(pars)#, *args, **kwargs)
    
    def log_probability(self, pars, *args, **kwargs):
        '''
        Log likelihood of the parameters, given the data
        p(pars | data) ~ p(pars) * p(data | pars)
        '''
        ln_prior = self.log_prior(pars)
        if not np.isfinite(ln_prior):
            return -np.inf
        return ln_prior + self.log_likelihood(pars)#, *args, **kwargs)


    def early_likelihood(self, pars, t_cut):
        """
            A likelihood for only the early model. Allows for fitting of nuisance parameters ahead of time. 

            Input:
                pars -- the early time model parameters.
                t_cut -- time limit out to which to fit early model to. Do not include plateau phase data for a good fit. 

            Returns:
                The likelihood. 
        """
        lkl = 0 
        if not self.model.rise:
            if self.model.decay_type == "exp":
                log_L, tau_decay, log_T = pars[:]
            elif self.model.decay_type == "pl":
                log_L, t_fb, p, log_T = pars[:]
        else:
            if self.model.decay_type == "exp":
                log_L, tau_decay, t_peak, sigma, log_T = pars[:]
            elif self.model.decay_type == "pl":
                log_L, t_fb, p, t_peak, sigma, log_T = pars[:]

        
        for band in self.model.data.bands_UV:    
            t_band, lum_band, err_band = self.model.data.args_band[band]
            v_band = self.model.data.bands_freq[band]
            s = self.model.data.global_systematic + self.model.data.bands_systematic[band]
            ii = (t_band < t_cut)
            if not self.model.rise:
                if self.model.decay_type == "exp":
                    ems = self.model.decay_model(t_band[ii], log_L, tau_decay, log_T, v_band)
                elif  self.model.decay_type == "pl":
                    ems = self.model.decay_model(t_band[ii], log_L, t_fb, p, log_T, v_band)
            else:
                if self.model.decay_type == "exp":
                    ems = self.model.decay_model(t_band[ii], log_L, tau_decay, t_peak, log_T, v_band) + self.model.rise_model(t_band[ii], log_L, sigma, t_peak, log_T, v_band)
                elif  self.model.decay_type == "pl":
                    ems = self.model.decay_model(t_band[ii], log_L, t_fb, p, t_peak, log_T, v_band) + self.model.rise_model(t_band[ii], log_L, sigma, t_peak, log_T, v_band)
            # Finish calculating the luminosities from the models:
            L_band = ems

            # Calculate and add to the likelihood:
            diff = lum_band[ii] - L_band
            var_band = err_band[ii]**2.0 + (s * lum_band[ii])**2.0
            lkl += -0.5 * ( (diff**2)/var_band ).sum()
        
        return lkl 


    ######################################
    ## Finding best fit
    ######################################

    
    def get_early_model_pars(self, p0=None, t_cut=365.25):
        """
            Finds the best fitting early model parameters from the 
            early model likelihood. 

            Input:
                p0 -- an initial guess at the early time model parameters.
                t_cut -- time limit out to which to fit early model to. Do not include plateau phase data for a good fit. 

            Returns:
                Maximum likelihood estimation of the early time parameters. 
        """

        if not self.model.rise:
            if self.model.decay_type == "exp":
                def to_min(pars, ):
                    return -self.early_likelihood(pars, t_cut)
                if p0 is None:
                    p0 = [44, 50, 4.5]
                out = minimize(to_min, x0=p0, bounds=(self.model.default_bounds["log_L"], self.model.default_bounds["t_decay"], self.model.default_bounds["log_T"]))

            elif self.model.decay_type == "pl":
                def to_min(pars, ):
                    return -self.early_likelihood(pars, t_cut)
                if p0 is None:
                    p0 = [44, 50, 2.0, 4.5]
                out = minimize(to_min, x0=p0, bounds=(self.model.default_bounds["log_L"], self.model.default_bounds["t_fb"], self.model.default_bounds["p"],self.model.default_bounds["log_T"]))
        else:
            if self.model.decay_type == "exp":
                def to_min(pars, ):
                    return -self.early_likelihood(pars, t_cut)
                if p0 is None:
                    p0 = [44, 50, 1.0, 10.0, 4.5]
                out = minimize(to_min, x0=p0, bounds=(self.model.default_bounds["log_L"], 
                                          self.model.default_bounds["t_decay"], self.model.default_bounds["t_peak"], 
                                          self.model.default_bounds["sigma"], self.model.default_bounds["log_T"]))

            elif self.model.decay_type == "pl":
                def to_min(pars, ):
                    return -self.early_likelihood(pars, t_cut)
                if p0 is None:
                    p0 = [44, 50, 2.0, 1.0, 10.0, 4.5]
                out = minimize(to_min, x0=p0, bounds=(self.model.default_bounds["log_L"], self.model.default_bounds["t_fb"], self.model.default_bounds["p"], 
                                          self.model.default_bounds["t_peak"], 
                                          self.model.default_bounds["sigma"],
                                          self.model.default_bounds["log_T"]))

        params = out.x
        
        return params
        
    @property
    def last_best_fit_pars(self):
        """
            Current best fit parameters of the fitting procedure. 
        """
        try:
            pars = self.last_best_fit.x.copy()
        except AttributeError:
            raise AttributeError("No best fit yet")
        return pars

    def best_fit(self, init_guess, print_best_fit=True, options=None, method='Nelder-Mead'):
        """
            Finds the best fitting model parameters from the total model likelihood. 

            Input:
                init_guess -- an initial guess of the model parameters.
                print_best_fit -- boolean. If true, prints output.
            
            Input (scipy.minimize): See scipy for more (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
                options=None
                method='Nelder-Mead'

            Returns:
                Maximum likelihood estimation of the system parameters. 

            Notes:
                This almost never works. Maximum likelihood estimation is not very robust for large parameter spaces. 
                Can help start a chain from a reasonable point, however. 
        """

        init_guess = np.array(init_guess).copy()

        def to_minimize(pars):
            return -self.log_probability(pars)
        
        out = minimize(to_minimize, x0=init_guess, bounds=self._log_prior.as_bounds(), method=method, options=options)
        
        if print_best_fit == True:
            log_mh, a_bh, m_d, r0, tvi, t0, incl = out.x[:7]
            if not self.model.rise:
                if self.model.decay_type=='exp':
                    log_L, t_decay, log_T = out.x[7:]
                elif self.model.decay_type=='pl':
                    log_L, t_decay, p, log_T = out.x[7:]
            else:
                if self.model.decay_type=='exp':
                    log_L, t_decay, t_peak, sigma, log_T = out.x[7:]
                elif self.model.decay_type=='pl':
                    log_L, t_decay, p, t_peak, sigma, log_T = out.x[7:]

            if print_best_fit:
                print(" ")
                print(" The fit {:s} successful".format(("was" if out.success else "was NOT")) )
                print(" scipy message",out.message)######for debugging
                print(" The best fitting parameters are: ")
                print(" log_10 Black hole mass = {:.3f}".format(log_mh)," solar masses",flush=True)
                print(" Black hole spin = {:.3f}".format(a_bh),flush=True)
                print(" Disc mass = {:.3f}".format(m_d)," solar masses",flush=True)
                print(" Initial radius = {:.2f}".format(r0)," r_g",flush=True)
                print(" Viscous timescale = {:.1f}".format(tvi)," days",flush=True)
                print(" Time offset = {:.1f}".format(t0)," days",flush=True)
                print(" Inclination angle = {:.1f}".format(incl)," degrees",flush=True)
                print(" ")
                print(" ")
                print(" The early time parameters are: ")
                print(" The early time luminosity is: {:.3f}".format(log_L), " log_10(erg/s) at 6e14 Hz. ")
                if self.model.decay_type=='exp':
                    print(" The early time decay rate is: {:.3f}".format(t_decay), " days. ")
                elif self.model.decay_type=='pl':
                    print(" The fallback time is: {:.3f}".format(t_decay), " days. ")
                    print(" The powerlaw index is: {:.3f}".format(p), ". ")
                if self.model.rise:
                    print(" The peak time is: {:.3f}".format(t_peak), " days. ")
                    print(" The rise time is: {:.3f}".format(sigma), " days. ")
                print(" The early time temperature is: {:.3f}".format(log_T), " log_10(Kelvin). ")
                print(" ")
        
        if out.success:
            self.last_best_fit = out
            return self.last_best_fit_pars
        
        # If failed still return 
        self.last_best_fit = out
        return self.last_best_fit_pars

        
    ######################################
    ## MCMC time
    ######################################
    def run_chain(self, 
                        full_pars=None, init_key_pars=None,
                        scatter=1e-3, nwalkers=100, nsteps=1000, 
                        progress=True, 
                        backend_path=None, overwrite_backend=False, backend_name="mcmc",
                        simple_chain_save=None,
                        p0=None, 
                        pool=None, pos = None,
                        **kwargs):
        """
            Run a MCMC chain to find best fitting parameters, using emcee (https://emcee.readthedocs.io/en/stable/). 

            Input:
                full_pars -- an initial guess at the model parameters.
                init_key_pars -- an initial guess at the disc model parameters. (will be ignored if full_pars specified). 
                p0 -- an initial guess at the early model parameters, if init_key_pars specified. 
            
            Input (emcee): See emcee for more (https://emcee.readthedocs.io/en/stable/). 
                scatter -- initial fractional scatter in walkers. 
                nwalkers -- number of walkers
                nsteps -- number of steps
                pos -- initial walker position (if None calculates internally). 

            Input (saving/backup):
                backend_path -- name of path to save backend to. 
                overwrite_backend -- boolean. If true, and backend exists, will overwrite. 
                simple_chain_save -- name of numpy file to save final chain to. 

            Returns:
                Saves a chain. 
        """
        if full_pars is not None or pos is not None:
            pass                
        elif (init_key_pars is None) and (self.last_best_fit is None):
            raise NoFitYetError("Please find a best fit first, or provide inital key parameters or initial full parametrs")
        elif init_key_pars is not None:
            if p0 is None:
                if not self.model.rise:
                    if self.model.decay_type == "exp":
                        p0 = [44, 50, 4.5]
                    elif self.model.decay_type == "pl":
                        p0 = [44, 50, 2.0, 4.5]
                else:
                    if self.model.decay_type == "exp":
                        p0 = [44, 50, 1.0, 10.0, 4.5]
                    elif self.model.decay_type == "pl":
                        p0 = [44, 50, 2.0, 1.0, 10.0, 4.5]
            full_pars = [*init_key_pars] + [*self.get_early_model_pars(p0=p0)]
        else:
            full_pars = self.last_best_fit.x
        
        # Double check that the starting priors don't return a infitine prior
        if not np.isfinite( self.log_prior(full_pars) ):
            raise ValueError("Initial parameters must be acceptable to the prior", full_pars)
        
        # Number of parameters
        ndim = len(full_pars)
        
        # If we need to set a backend, create it
        if backend_path is not None:
            if backend_path[-3:] != ".h5":
                backend_path += ".h5"
            if os.path.exists(backend_path) and overwrite_backend:
                print("Existing %s has been deleted" % backend_path, flush=True)
                os.remove(backend_path)
            backend = emcee.backends.HDFBackend(backend_path, name=backend_name)
            if overwrite_backend:
                backend.reset(nwalkers, ndim)
            else:
                try:
                    pos = backend.get_last_sample()
                except AttributeError as e:
                    pass
        else:
            backend=None
        
        if pos is None:
            # Trying to fix edge cases... 
            full_pars = np.asarray(full_pars)
            ii_zero = abs(full_pars) < 1e-4#Sometimes Schwarzschild black hole initialised, wrecks algorithm. 
            full_pars[ii_zero] += 1e-2
            pos = full_pars * (1 + scatter * np.random.randn(nwalkers, ndim) )
            
        
        if pool is not None:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, kwargs=kwargs,
                                            backend=backend, pool=pool)
        else:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, kwargs=kwargs,
                                            backend=backend)

        # Run chain
        self.sampler.run_mcmc(pos, nsteps, progress=progress, skip_initial_state_check=True)  # Progress prints out progress bar
        
        # Get chain
        chain = self.sampler.get_chain()
        chain_probs=self.sampler.get_log_prob()

        self.chain = chain
        self.chain_probabilities = chain_probs
        
        # Save chain as numpy file
        if simple_chain_save is not None:
            np.save(simple_chain_save, chain)
            np.save(simple_chain_save+'-probs', chain_probs)
        
        return 

    # def run_nested_sampling(self,log_dir):
        
    #     # TO DO: add function to change the bounds for the uniform prior
    #     # warmstart_from_similar_file may also be useful if you have already run a different model
    #     # If this crashes, add resume = True to the sampler as well as the log_dir for the run
        
    #     # Transforms a given prior with values from 0 to 1 to our physical scales:
    #     def transform(quantile_cube):
    #         lowers=self._log_prior.as_bounds()[:,0]
    #         uppers=self._log_prior.as_bounds()[:,1]
    #         return lowers + (uppers-lowers) * quantile_cube

    #     param_names=self.model.default_key_pars+self.model.default_early_pars

    #     sampler = ultranest.ReactiveNestedSampler(param_names, self.log_likelihood, transform, log_dir=log_dir)

    #     nsteps = 2 * len(param_names)
    #     # Adding a step sampler since our models have many parameters:
    #     sampler.stepsampler = ultranest.stepsampler.SliceSampler(
    #         nsteps=nsteps,
    #         generate_direction=ultranest.stepsampler.generate_mixture_random_direction
    #     )

    #     # Running the sampler, will save as it goes
    #     result2 = sampler.run(frac_remain=0.5)

    #     sampler.print_results()

