import os
import numpy as np
import emcee
import pickle
import multiprocessing
import matplotlib.pyplot as plt 
import corner
from scipy.optimize import minimize
from . import prior as pr
from .models import *
from .constants import *


__all__ = ["NoFitYetError", "Fit"]


class NoFitYetError(Exception):
    pass

# Module-level variables for multiprocessing workers
_worker_model = None
_worker_prior = None

def _init_worker(model, prior):
    """Initialize worker process with model and prior.
    
    This function is called once per worker process when the pool is created.
    It sets up the model and prior in module-level variables so they can be
    accessed by the worker function without pickling self.
    """
    global _worker_model, _worker_prior
    _worker_model = model
    _worker_prior = prior

def _worker_log_probability(pars):
    """Worker function for multiprocessing - uses module-level variables.
    
    This function is called by emcee to evaluate log-probability in worker
    processes. It uses module-level variables set by _init_worker, avoiding
    the need to pickle self or the model on every evaluation.
    """
    global _worker_model, _worker_prior
    if _worker_model is None or _worker_prior is None:
        raise RuntimeError("Worker not initialized - _init_worker must be called first")
    
    ln_prior = _worker_prior(pars)
    if not np.isfinite(ln_prior):
        return -np.inf
    return ln_prior + _worker_model.log_likelihood(pars)

# LaTeX labels for the seven key (disc) parameters.  These are the only ones
# that change name under the fit_log_* / fit_cos_incl reparameterisations, so
# the plotting routines patch just these and leave the early/rise labels alone.
_KEY_PAR_LABELS = {
    "log_mh":     r"$\log M_\bullet$",
    "a_bh":       r"$a_\bullet$",
    "m_disc":     r"$M_{\rm disc}$",
    "log_m_disc": r"$\log M_{\rm disc}$",
    "r0":         r"$r_0$",
    "log_r0":     r"$\log r_0$",
    "tvi":        r"$t_{\rm visc}$",
    "log_tvi":    r"$\log t_{\rm visc}$",
    "t0":         r"$t_0$",
    "incl":       r"$i$",
    "cos_incl":   r"$\cos i$",
}


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
        # Optional: names for parameters in the chain (used for robust conversions when sampling).
        # When present, enables auto-detection of whether the chain used m_disc or log_m_disc.
        self._chain_param_names = None

    ######################################
    ## Saving and loading 
    ######################################

    def save(self, name):
        """save class as name_FIT.pickle"""
        file = open(name+'_FIT.pickle','wb')
        file.write(pickle.dumps(self.__dict__))
        file.close() 

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

    ######################################
    ## Plotting data, models and analysis
    ######################################
    
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


    def _label_key_pars(self, labels):
        """Relabel the key parameters using the names the chain actually holds.

        A chain fit with fit_log_m_disc / fit_log_r0 / fit_log_tvi / fit_cos_incl
        stores log10(m_disc), log10(r0), log10(tvi) and cos(incl).  Without this
        the hardcoded labels would say "M_disc" and "i" over log and cosine
        values -- a silently wrong figure.  When the chain holds the linear
        parameters (or predates _chain_param_names) the labels are unchanged.
        """
        names = getattr(self, "_chain_param_names", None)
        if not names:
            return labels
        out = list(labels)
        for i, nm in enumerate(list(names)[:7]):
            if i < len(out) and nm in _KEY_PAR_LABELS:
                out[i] = _KEY_PAR_LABELS[nm]
        return out

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
            labels = self._label_key_pars(labels)
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
            labels = self._label_key_pars(labels)
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
        
        labels = self._label_key_pars(labels)
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
                                as_range=True, plotted_range=[16, 84], 
                                ignore_sigma=3, 
                                show_disc=True, show_early=False, show_total=True, 
                                save_posterior=False, save_name='save_lightcurves.txt'):
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
            sample = flat_samples[-j]
            if not m.rise:
                if m.decay:
                    if m.decay_type == 'exp':
                        converted = m.convert_parameters(sample, param_names=getattr(self, "_chain_param_names", None)) if hasattr(m, "convert_parameters") else sample
                        log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, log_T = converted
                    elif m.decay_type == 'pl':
                        converted = m.convert_parameters(sample, param_names=getattr(self, "_chain_param_names", None)) if hasattr(m, "convert_parameters") else sample
                        log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, p, log_T = converted
                else:
                    converted = m.convert_parameters(sample, param_names=getattr(self, "_chain_param_names", None)) if hasattr(m, "convert_parameters") else sample
                    log_mh, a_bh, m_disc, r0, tvi, t0, incl = converted
            else:
                if m.decay:
                    if m.decay_type == 'exp':
                        converted = m.convert_parameters(sample, param_names=getattr(self, "_chain_param_names", None)) if hasattr(m, "convert_parameters") else sample
                        log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, t_peak, sigma, log_T = converted
                    elif m.decay_type == 'pl':
                        converted = m.convert_parameters(sample, param_names=getattr(self, "_chain_param_names", None)) if hasattr(m, "convert_parameters") else sample
                        log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, p, t_peak, sigma, log_T = converted
                else:
                    converted = m.convert_parameters(sample, param_names=getattr(self, "_chain_param_names", None)) if hasattr(m, "convert_parameters") else sample
                    log_mh, a_bh, m_disc, r0, tvi, t0, incl = converted
                    


            # Sigma-clipping is done in the *native chain parameterization* (linear or log),
            # i.e. before any conversion of disc mass.
            delta =  abs((sample-medians)/stds)
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


        if save_posterior:
            for ii, band in enumerate(bands): 
                post_m = np.percentile(lms[:, :, ii], plotted_range, axis=1)
                post_e = np.percentile(les[:, :, ii], plotted_range, axis=1)
                post_m_ = np.percentile(lms[:, :, ii], [50], axis=1)
                post_e_ = np.percentile(les[:, :, ii], [50], axis=1)

                with open('./%s_%s'%(band, save_name), 'w') as file:
                    file.write('###Time[days]\tMedian Disk Lum\t+range Disk Lum\t-range Disk Lum\tMedian Early Lum\t+range Early Lum\t-range Early Lum\n')

                    for k in range(len(t_plot)):
                        file.write(f'{t_plot[k]}\t{post_m_[0][k]}\t{post_m[1][k]}\t{post_m[0][k]}\t{post_e_[0][k]}\t{post_e[1][k]}\t{post_e[0][k]}\n')

                file.close()

        return fig


    def plot_bolometric_luminosity_posterior(self, t_plot, fig=None, as_eddington=False, 
                                N=100, f_discard=0.5, plot_median=True, 
                                as_range=True, plotted_range=[16, 84], 
                                ignore_sigma=3, yscale='log', xscale='log', 
                                ylabel=r'$L_{\rm bol}$ [erg/s]', xlabel=r'Time [days]', 
                                ylim=None, xlim=None, color='k', alpha=0.3,
                                save_posterior=False, save_name='save_bolometric.txt'):
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
            sample = flat_samples[-j]
            delta =  abs((sample[:6]-medians[:6])/stds[:6])
            if (delta>ignore_sigma).any():
                pass
            else:
                converted = m.convert_parameters(sample, param_names=getattr(self, "_chain_param_names", None)) if hasattr(m, "convert_parameters") else sample
                log_mh, a_bh, m_disc, r0, tvi, t0 = converted[:6]
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

        if save_posterior:
            with open('./%s'%save_name, 'w') as file:
                file.write('###Time[days]\tMedian Lbol\t+range Lbol\t-range Lbol\n')

                for k in range(len(t_plot)):
                    file.write(f'{t_plot[k]}\t{med[0][k]}\t{ds[1][k]}\t{ds[0][k]}\n')

            file.close()

    
        return fig 


    def plot_isco_accretion_rate_posterior(self, t_plot, fig=None, as_eddington=True, 
                                N=100, f_discard=0.5, plot_median=True, 
                                as_range=True, plotted_range=[16, 84], 
                                ignore_sigma=3, yscale='log', xscale='log', 
                                ylabel=r'$\dot M_{\rm acc}(r_I)$ [g/s]', xlabel=r'Time [days]', 
                                ylim=None, xlim=None, color='k', alpha=0.3, 
                                save_posterior=False, save_name='save_mdot.txt'):
        r"""
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
            sample = flat_samples[-j]
            delta =  abs((sample[:6]-medians[:6])/stds[:6])
            if (delta>ignore_sigma).any():
                pass
            else:
                converted = m.convert_parameters(sample, param_names=getattr(self, "_chain_param_names", None)) if hasattr(m, "convert_parameters") else sample
                log_mh, a_bh, m_disc, r0, tvi, t0 = converted[:6]
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

        if save_posterior:
            with open('./%s'%save_name, 'w') as file:
                file.write('###Time[days]\tMedian Mdot\t+range Mdot\t-range Mdot\n')

                for k in range(len(t_plot)):
                    file.write(f'{t_plot[k]}\t{med[0][k]}\t{ds[1][k]}\t{ds[0][k]}\n')

            file.close()

    
        return fig 


    def plot_density_posterior(self, t_plot, fig=None, in_cgs=False, 
                                N=100, f_discard=0.5, plot_median=True, 
                                as_range=True, plotted_range=[16, 84], 
                                ignore_sigma=3, yscale='log', xscale='log', 
                                ylabel=r'$\Sigma$ [g/cm$^2$]', xlabel=r'$r/r_g$', 
                                ylim=None, xlim=None, colors=['b'], alpha=0.3, 
                                rmin=6, rmax=2000, legend=True):
        r"""
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
        
        # Ensure colors list is long enough (cycle through if needed)
        if len(colors) < len(t_plot):
            # Repeat colors to match length of t_plot
            colors = (colors * ((len(t_plot) // len(colors)) + 1))[:len(t_plot)]

        ss = np.zeros(len(t_plot)*N*Nr).reshape(len(t_plot), Nr, N)
        rr = np.zeros(N*Nr).reshape(Nr, N)
        rr2 = np.zeros(N*Nr).reshape(Nr, N)

        k=0

        while k < N:
            j = np.random.randint(len(flat_samples))
            sample = flat_samples[-j]
            delta =  abs((sample[:6]-medians[:6])/stds[:6])
            if (delta>ignore_sigma).any():
                pass
            else:
                converted = m.convert_parameters(sample, param_names=getattr(self, "_chain_param_names", None)) if hasattr(m, "convert_parameters") else sample
                log_mh, a_bh, m_disc, r0, tvi, t0 = converted[:6]
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
        
        if print_best_fit:
            # Report PHYSICAL values, whatever space the fit was performed in.
            #
            # This used to unpack out.x positionally and print it under physical
            # labels.  With the v2.0 defaults -- log_m_disc, log_r0, log_tvi,
            # cos_incl -- that printed things like "Disc mass = -0.242 solar
            # masses" and "Inclination angle = 0.3 degrees".  It also assumed the
            # early-time parameters always exist, so it raised a ValueError for
            # any model built with decay=False.
            key_names = list(self.model.default_key_pars)
            early_names = list(getattr(self.model, 'default_early_pars', []))
            names = key_names + early_names
            phys = self.model.convert_parameters(out.x, names)

            physical_key = dict(zip(
                ["log_mh", "a_bh", "m_disc", "r0", "tvi", "t0", "incl"], phys[:7]))
            early = dict(zip(early_names, phys[len(key_names):]))

            print(" ")
            print(" The fit {:s} successful".format("was" if out.success else "was NOT"))
            print(" scipy message", out.message)
            print(" The best fitting parameters are: ")
            print(" log_10 Black hole mass = {:.3f}".format(physical_key["log_mh"]),
                  " solar masses", flush=True)
            print(" Black hole spin = {:.3f}".format(physical_key["a_bh"]), flush=True)
            print(" Disc mass = {:.4g}".format(physical_key["m_disc"]),
                  " solar masses", flush=True)
            print(" Initial radius = {:.2f}".format(physical_key["r0"]), " r_g", flush=True)
            print(" Viscous timescale = {:.1f}".format(physical_key["tvi"]), " days", flush=True)
            print(" Time offset = {:.1f}".format(physical_key["t0"]), " days", flush=True)
            print(" Inclination angle = {:.1f}".format(physical_key["incl"]),
                  " degrees", flush=True)
            print(" ")

            if early_names:
                print(" The early time parameters are: ")
                if "log_L" in early:
                    print(" The early time luminosity is: {:.3f}".format(early["log_L"]),
                          " log_10(erg/s) at 6e14 Hz. ")
                if "t_decay" in early:
                    print(" The early time decay rate is: {:.3f}".format(early["t_decay"]), " days. ")
                if "t_fb" in early:
                    print(" The fallback time is: {:.3f}".format(early["t_fb"]), " days. ")
                if "p" in early:
                    print(" The powerlaw index is: {:.3f}".format(early["p"]), ". ")
                if "t_peak" in early:
                    print(" The peak time is: {:.3f}".format(early["t_peak"]), " days. ")
                if "sigma" in early:
                    print(" The rise time is: {:.3f}".format(early["sigma"]), " days. ")
                if "log_T" in early:
                    print(" The early time temperature is: {:.3f}".format(early["log_T"]),
                          " log_10(Kelvin). ")
                print(" ")


        if out.success:
            self.last_best_fit = out
            return self.last_best_fit_pars
        
        # If failed still return 
        self.last_best_fit = out
        return self.last_best_fit_pars

    def _check_backend_compression_support(self):
        """
        Check if emcee's HDFBackend supports compression.
        
        Returns
        -------
        supports_compression : bool
            True if compression is supported
        """
        try:
            import inspect
            sig = inspect.signature(emcee.backends.HDFBackend.__init__)
            return 'compression' in sig.parameters
        except Exception:
            return False

    def _compute_gelman_rubin(self, chain):
        """
        Compute Gelman-Rubin statistic (R-hat) for convergence checking.
        
        Parameters
        ----------
        chain : array
            MCMC chain of shape (nsteps, nwalkers, ndim)
        
        Returns
        -------
        rhat : array
            R-hat statistic for each parameter (ndim,)
        """
        nsteps, nwalkers, ndim = chain.shape
        
        if nsteps < 2:
            return np.full(ndim, np.inf)  # Not enough steps
        
        # Compute within-chain variance (W)
        # Mean for each walker
        walker_means = np.mean(chain, axis=0)  # (nwalkers, ndim)
        # Variance for each walker
        walker_vars = np.var(chain, axis=0, ddof=1)  # (nwalkers, ndim)
        # Average within-chain variance
        W = np.mean(walker_vars, axis=0)  # (ndim,)
        
        # Compute between-chain variance (B)
        # Overall mean for each parameter
        overall_means = np.mean(chain, axis=(0, 1))  # (ndim,)
        # Between-chain variance
        B = (nsteps / (nwalkers - 1)) * np.sum((walker_means - overall_means)**2, axis=0)  # (ndim,)
        
        # Compute pooled variance
        var_pooled = ((nsteps - 1) / nsteps) * W + (1 / nsteps) * B
        
        # R-hat
        rhat = np.sqrt(var_pooled / W)
        
        return rhat


    def compute_rhat(self, chain=None, f_discard=0.0):
        """
        Gelman-Rubin convergence statistic (R-hat) for each parameter.

        Treats each emcee walker as a chain.  Values below about 1.01 are the
        usual convergence criterion; well above that means the walkers have not
        yet forgotten where they started.

        Parameters
        ----------
        chain : array or None
            Chain of shape (nsteps, nwalkers, ndim).  Defaults to `self.chain`.
        f_discard : float
            Fraction of the chain to discard as burn-in before computing R-hat.
            Defaults to 0, i.e. use everything.

        Returns
        -------
        rhat : np.ndarray
            R-hat for each parameter, ordered as
            `model.default_key_pars + model.default_early_pars`.
        """
        if chain is None:
            chain = getattr(self, 'chain', None)
        if chain is None:
            raise ValueError("No chain available -- run run_chain() first, "
                             "or pass a chain explicitly.")
        chain = np.asarray(chain)
        if f_discard:
            chain = chain[int(f_discard * len(chain)):]
        return self._compute_gelman_rubin(chain)


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
                        use_parallel=True, n_workers=None,
                        moves=None, use_optimized_moves=True,
                        backend_compress=False, backend_compress_opts=None,
                        check_convergence=False, convergence_check_interval=100,
                        target_rhat=1.01, min_steps=500,
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

            Input (parallelization):
                use_parallel -- boolean. If True (default), automatically create a multiprocessing pool to use multiple cores.
                n_workers -- int or None. Number of worker processes. If None (default), uses min(nwalkers, cpu_count()).
                pool -- multiprocessing pool. If provided, uses this pool. If None and use_parallel=True, creates one automatically.

            Input (moves):
                moves -- emcee move object or list of (move, weight) tuples. 
                         If None and use_optimized_moves=True, uses optimized default mixture.
                use_optimized_moves -- boolean. If True (default) and moves=None, uses 
                                      optimized move mixture (StretchMove + DEMove + DESnookerMove).

            Input (backend optimization):
                backend_compress -- bool or str. If True, enables gzip compression. If a string 
                                   (e.g., 'gzip', 'lzf', 'szip'), uses that compression type. 
                                   Requires emcee >= 3.1. Default: False.
                backend_compress_opts -- dict or int. Compression options. For gzip, can be 
                                        an integer (compression level 1-9). Default: None (uses 
                                        default compression level 4 for gzip).

            Input (convergence checking):
                check_convergence -- boolean. If True, monitor Gelman-Rubin statistic and 
                                    stop early when converged. Default: False.
                convergence_check_interval -- int. Check convergence every N steps. Default: 100.
                target_rhat -- float. Target R-hat statistic for convergence. Default: 1.01.
                min_steps -- int. Minimum steps before checking convergence. Default: 500.

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

        # Record parameter names for this chain (used later for robust chain sampling / plotting).
        # This is especially important when models support alternative parameterizations
        # (e.g. m_disc vs log_m_disc).
        try:
            self._chain_param_names = list(self.model.default_key_pars + self.model.default_early_pars)
        except Exception:
            self._chain_param_names = None
        
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
            
            # Setup backend with optional compression
            backend_kwargs = {"name": backend_name}
            
            if backend_compress:
                # Check if compression is supported
                compression_supported = self._check_backend_compression_support()
                
                if compression_supported:
                    # Set compression type
                    if backend_compress is True:
                        # Default to gzip compression
                        compression_type = 'gzip'
                    else:
                        # User specified compression type (e.g., 'gzip', 'lzf', 'szip')
                        compression_type = backend_compress
                    
                    backend_kwargs["compression"] = compression_type
                    
                    # Add compression options if provided
                    if backend_compress_opts is not None:
                        backend_kwargs["compression_opts"] = backend_compress_opts
                    elif compression_type == 'gzip':
                        # Default gzip compression level (1-9, 4 is a good balance)
                        backend_kwargs["compression_opts"] = 4
                    
                    if progress:
                        print(f"Using HDF5 compression: {compression_type}", flush=True)
                else:
                    # Compression not supported by this emcee version
                    import warnings
                    warnings.warn("Backend compression requested but not supported by this emcee version. "
                                "Compression disabled. Upgrade to emcee >= 3.1 for compression support.")
            
            backend = emcee.backends.HDFBackend(backend_path, **backend_kwargs)
            
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
            
        # Create pool if needed for parallelization
        pool_created = False
        if pool is None and use_parallel:
            try:
                # Try to set 'spawn' start method (works best with emcee)
                # On macOS, use force=True to override if already set
                # On Linux/other, only set if not already set
                import platform
                try:
                    if platform.system() == 'Darwin':
                        multiprocessing.set_start_method('spawn', force=True)
                    else:
                        # Only set if not already set
                        current_method = multiprocessing.get_start_method(allow_none=True)
                        if current_method is None:
                            multiprocessing.set_start_method('spawn')
                except RuntimeError:
                    # Start method already set, or cannot be set (e.g., missing if __name__ == '__main__' guard)
                    pass
                
                if n_workers is None:
                    n_workers = min(nwalkers, multiprocessing.cpu_count())
                
                # Create pool with initializer to set up workers once
                # This avoids pickling the model on every evaluation
                pool = multiprocessing.Pool(
                    n_workers,
                    initializer=_init_worker,
                    initargs=(self.model, self._log_prior)
                )
                pool_created = True
                
                # Warn about NumPy threading if not disabled (helpful for performance)
                if progress and os.environ.get('OMP_NUM_THREADS', '') != '1':
                    import warnings
                    warnings.warn("For optimal parallelization performance, set OMP_NUM_THREADS=1 "
                                "to avoid conflicts between NumPy threading and multiprocessing. "
                                "Run: export OMP_NUM_THREADS=1")
            except Exception as e:
                # If pool creation fails (e.g., missing if __name__ == '__main__' guard),
                # provide helpful error message and fall back to serial
                error_msg = str(e)
                if "bootstrapping phase" in error_msg or "if __name__ == '__main__'" in error_msg:
                    import warnings
                    warnings.warn("Parallel execution requires 'if __name__ == \"__main__\":' guard in your script. "
                                "Wrap your code in a main() function and call it with 'if __name__ == \"__main__\": main()'. "
                                "Falling back to serial execution. See documentation for details.")
                if progress:
                    print(f"Failed to create parallel pool: {e}. Continuing with serial execution.", flush=True)
                pool = None
        
        # Use worker function for multiprocessing, regular method for serial
        # The worker function uses module-level variables set by _init_worker,
        # avoiding the need to pickle self or model on every evaluation
        if pool is not None:
            log_prob_func = _worker_log_probability
            if progress:
                print(f"Parallel execution enabled with {n_workers} workers", flush=True)
        else:
            log_prob_func = self.log_probability
            if progress and use_parallel:
                print(f"Serial execution (parallel requested but pool unavailable)", flush=True)
        
        # Setup moves
        moves_to_use = None
        if moves is not None:
            # User specified moves
            moves_to_use = moves
        elif use_optimized_moves:
            # Use optimized default mixture
            try:
                from emcee import moves as emcee_moves
                moves_to_use = [
                    (emcee_moves.StretchMove(a=2.0), 0.5),
                    (emcee_moves.DEMove(), 0.3),
                    (emcee_moves.DESnookerMove(), 0.2)
                ]
                if progress:
                    print("Using optimized move mixture (StretchMove + DEMove + DESnookerMove)", flush=True)
            except (ImportError, AttributeError) as e:
                # emcee < 3.0 or moves not available, use default
                import warnings
                warnings.warn(f"Optimized moves not available ({e}). Using emcee defaults.")
                moves_to_use = None
        # else: moves_to_use = None (use emcee defaults)
        
        try:
            if pool is not None:
                self.sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_func, kwargs=kwargs,
                                                    backend=backend, pool=pool, moves=moves_to_use)
            else:
                self.sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_func, kwargs=kwargs,
                                                backend=backend, moves=moves_to_use)

            # Run chain with diagnostics
            if pool is not None and progress:
                # Measure time per likelihood evaluation for diagnostics
                # Use self.log_probability for timing since we're in the main process
                import time
                test_start = time.perf_counter()
                # Use full_pars for test (guaranteed to be 1D) instead of pos[0]
                test_pars = np.asarray(full_pars).flatten()[:ndim]  # Ensure 1D array
                _ = self.log_probability(test_pars)  # Test single evaluation in main process
                test_time = time.perf_counter() - test_start
                if test_time > 0:
                    print(f"Estimated time per likelihood evaluation: {test_time*1000:.2f} ms", flush=True)
                    if test_time < 0.1:  # Less than 100ms
                        import warnings
                        warnings.warn("Likelihood evaluations are very fast (< 100ms). "
                                    "Parallelization overhead may dominate. Consider using more steps "
                                    "or a more expensive model to see speedup benefits.")
            
            # Run chain with optional convergence checking
            if check_convergence:
                # Run in chunks, checking convergence periodically
                steps_completed = 0
                remaining_steps = nsteps
                
                # Ensure we have minimum steps before checking
                if nsteps < min_steps:
                    # Run all steps without checking
                    self.sampler.run_mcmc(pos, nsteps, progress=progress, skip_initial_state_check=True)
                    steps_completed = nsteps
                else:
                    # Run initial steps without checking
                    initial_steps = min_steps
                    if initial_steps > 0:
                        self.sampler.run_mcmc(pos, initial_steps, progress=False, skip_initial_state_check=True)
                        steps_completed += initial_steps
                        pos = self.sampler.get_last_sample()
                        remaining_steps -= initial_steps
                    
                    # Run remaining steps with convergence checking
                    converged = False
                    while remaining_steps > 0 and not converged:
                        # Run next chunk
                        chunk_steps = min(convergence_check_interval, remaining_steps)
                        self.sampler.run_mcmc(pos, chunk_steps, progress=False, skip_initial_state_check=True)
                        steps_completed += chunk_steps
                        remaining_steps -= chunk_steps
                        
                        # Check convergence
                        chain = self.sampler.get_chain()
                        rhat = self._compute_gelman_rubin(chain)
                        max_rhat = np.max(rhat)
                        
                        if progress:
                            print(f"Step {steps_completed}/{nsteps}: max R-hat = {max_rhat:.4f}", flush=True)
                        
                        if max_rhat < target_rhat:
                            converged = True
                            if progress:
                                print(f"Chain converged at step {steps_completed} (max R-hat = {max_rhat:.4f} < {target_rhat})", flush=True)
                            break
                        
                        pos = self.sampler.get_last_sample()
                    
                    # If we stopped early, update progress display
                    if converged and progress:
                        print(f"Early stopping: converged after {steps_completed} steps (planned: {nsteps})", flush=True)
            else:
                # Run normally without convergence checking
                self.sampler.run_mcmc(pos, nsteps, progress=progress, skip_initial_state_check=True)
        finally:
            # Clean up pool if we created it
            if pool_created and pool is not None:
                pool.close()
                pool.join()
        
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

