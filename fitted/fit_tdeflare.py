import os
import numpy as np
import emcee
import pickle
import matplotlib.pyplot as plt 
import corner
from scipy.optimize import minimize
from . import prior as pr
from .models import *
from .constants import *
from warnings import warn

__all__ = ["NoFitYetError", "FitTDEFLARE"]


class NoFitYetError(Exception):
    pass

class FitTDEFLARE():
    def __init__(self, model=None, prior=None):
        """
            FitTeD Fit class. The class which controls statistical analysis of the data, and fitting procedures. 
            Fit class also contains all of the analysis plotting methods. 

            Inputs:
                model -- FitTeD model class. If None, defaults to GR_disc with zero data. 
                prior -- input parameter prior. If none, defaults to window prior.
        """
        if model is None:
            self.model = TDEFLARE(data=None, 
              rise=False, rise_type='gauss')
        elif isinstance(model, TDEFLARE):
            self.model = model
        else:
            raise TypeError("Model must be of type " + str(TDEFLARE) )
        
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


    def plot_corner(self, f_discard=0.5, fig=None, color=None):#... other options. 
        """
            Plots emcee output as a corner plot.

            Input:
                f_discard -- factor f, 0<f<1, fraction of chain to discard for burn in. 
            
            Input (matplotlib):
                All standard stuff:
                    fig=None, 
                    color=None. 

            Returns -- fig; matplotlib figure class.
        """
        
        samples = self.chain
        samples[:, :, 3] = np.log10(10**samples[:, :, 2] * samples[:, :, 3] * 60 * 60 * 24)## from time to energy
        ndim = len(samples[0, 0])
        n_discard = int(len(samples) * f_discard)
        flat_samples = samples[n_discard:].reshape((-1, ndim))
        if not self.model.rise:
            if self.model.decay_type == 'exp':
                labels=[r"$\log L_P$", r"$\log T_P$", r"$\log L_{\rm pk}$", r"$\log E_{g}$", r"$\log T$"]
            elif self.model.decay_type == 'pl':
                labels=[]
        else:
            if self.model.decay_type == 'exp':
                labels=[r"$\log L_P$", r"$\log T_P$", r"$\log L_{\rm pk}$", r"$\log E_{g}$", r"$t_{\rm peak}$", r"$\sigma_{\rm rise}$", r"$\log T$"]
            elif self.model.decay_type == 'pl':
                labels=[]
        fig = corner.corner(flat_samples, color=color, plot_contours=True, plot_datapoints=False, smooth=True, labels=labels, density=True, fig=fig)
        
        return fig


    def plot_walkers(self, fig=None, color='k', alpha=0.3):#... other options. 
        """
            Plots emcee walkers.
            
            Input (matplotlib):
                All standard stuff:
                    fig=None, 
                    color='k',
                    alpha=0.3.  

            Returns -- fig; matplotlib figure class.
        """
        
        samples = self.chain
        samples[:, :, 3] = np.log10(10**samples[:, :, 2] * samples[:, :, 3] * 60 * 60 * 24)## from time to energy
        if not self.model.rise:
            if self.model.decay_type == 'exp':
                labels=[r"$\log L_P$", r"$\log T_P$", r"$\log L_{\rm pk}$", r"$\log E_{g}$", r"$\log T$"]
            elif self.model.decay_type == 'pl':
                labels=[]
        else:
            if self.model.decay_type == 'exp':
                labels=[r"$\log L_P$", r"$\log T_P$", r"$\log L_{\rm pk}$", r"$\log E_{g}$", r"$t_{\rm peak}$", r"$\sigma_{\rm rise}$", r"$\log T$"]
            elif self.model.decay_type == 'pl':
                labels=[]

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
            if not m.rise:
                if m.decay:
                    if m.decay_type == 'exp':
                        logP, logTp, log_L, tdecay, log_T = flat_samples[-j]
                        # tdecay = 10**(logEg - log_L - np.log10(60*60*24))
                        t_peak = 0
                    elif m.decay_type == 'pl':
                        pass
                else:
                    pass
            else:
                if m.decay:
                    if m.decay_type == 'exp':
                        logP, logTp, log_L, tdecay, t_peak, sigma, log_T = flat_samples[-j]
                        # tdecay = 10**(logEg - log_L - np.log10(60*60*24))
                    elif m.decay_type == 'pl':
                        pass
                else:
                    pass
                    


            delta =  abs((flat_samples[-j]-medians)/stds)
            if (delta>ignore_sigma).any():
                pass
            else:
                for ii, band in enumerate(bands):
                    if band in self.model.data.bands_UV:
                        lmod = m.model_UV(t_plot, t_peak, logP, logTp, v=d.bands_freq[band])
                        if not m.rise:
                            if m.decay:
                                if m.decay_type == 'exp':
                                    emod = m.decay_model(t_plot, log_L, tdecay, log_T, v=d.bands_freq[band])
                                elif m.decay_type == 'pl':
                                    pass
                            else:
                                pass
                        else:
                            if m.decay:
                                if m.decay_type == 'exp':
                                    emod = m.decay_model(t_plot, log_L, tdecay, t_peak, log_T, v=d.bands_freq[band]) + m.rise_model(t_plot, log_L, sigma, t_peak, log_T, v=d.bands_freq[band])
                                elif m.decay_type == 'pl':
                                    pass
                            else:
                                pass
                        if np.isnan(lmod).any():
                            pass
                        else:
                            lms[:, k, ii] = lmod
                            les[:, k, ii] = emod
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
            logP, logTp = out.x[:2]
            if not self.model.rise:
                if self.model.decay_type=='exp':
                    log_L, t_decay, log_T = out.x[2:]
                elif self.model.decay_type=='pl':
                    log_L, t_decay, p, log_T = out.x[2:]
            else:
                if self.model.decay_type=='exp':
                    log_L, t_decay, t_peak, sigma, log_T = out.x[2:]
                elif self.model.decay_type=='pl':
                    log_L, t_decay, p, t_peak, sigma, log_T = out.x[2:]

            if print_best_fit:
                print(" ")
                print(" The fit {:s} successful".format(("was" if out.success else "was NOT")) )
                print(" scipy message",out.message)######for debugging
                print(" The best fitting parameters are: ")
                print(" log_10 Plateau luminosity = {:.3f}".format(logP)," erg/s at 6e14 Hz.",flush=True)
                print(" log_10 Plateau temperature = {:.1f}".format(logTp)," erg/s",flush=True)
                print(" ")
                print(" ")
                print(" The early time parameters are: ")
                print(" The early time luminosity is: {:.3f}".format(log_L), " log_10(erg/s) at 6e14 Hz. ")
                if self.model.decay_type=='exp':
                    print(" The early time decay rate is: {:.3f}".format(t_decay), " days. ")
                    print(" The early time g-band energy is: {:.3f}".format(np.log10(t_decay*10**log_L*60*60*24)), " erg. ")
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

    ######################################
    ## Processing fit
    ######################################
    def get_mass_constraints(self, f_discard=0.5, do_hills=True):

        def mass_peak(log_masses, Lpeak, epeak):
            eps_Lp = (0.53**2.0 + epeak**2.0)**0.5
            alpha_Lp = 6.52
            beta_Lp = 0.98
            log_M0 = alpha_Lp + beta_Lp * (Lpeak-43)
            return 1/(2*np.pi*eps_Lp**2)**0.5 * np.exp(-(log_M0-log_masses)**2/(2*eps_Lp**2))

        def mass_energy(log_masses, Eg, eg):
            eps_Ep = (0.44**2.0 + eg**2.0)**0.5
            alpha_Ep = 6.78
            beta_Ep = 0.98
            log_M0 = alpha_Ep + beta_Ep * (Eg - 50)
            return 1/(2*np.pi*eps_Ep**2)**0.5 * np.exp(-(log_M0-log_masses)**2/(2*eps_Ep**2))

        def mass_plat(log_masses, P, eP):
            eps_P =  (0.38**2.0 + eP**2.0)**0.5
            alpha_P = 9.0
            beta_P = 1.50
            log_M0 = alpha_P + beta_P * (P - 43)
            return 1/(2*np.pi*eps_P**2)**0.5 * np.exp(-(log_M0-log_masses)**2/(2*eps_P**2))
        
        samples = self.chain
        samples[:, :, 3] = np.log10(10**samples[:, :, 2] * samples[:, :, 3] * 60 * 60 * 24)## from time to energy
        ndim = len(samples[0, 0])
        n_discard = int(len(samples) * f_discard)
        flat_samples = samples[n_discard:].reshape((-1, ndim))        

        post_m = np.median(flat_samples, axis=0)
        post_e = np.std(flat_samples, axis=0)

        P = post_m[0]
        e_P = post_e[0]
        L_pk = post_m[2]
        e_pk = post_e[2]
        Eg = post_m[3]
        e_eg = post_e[3]

        log_masses = np.linspace(4, 10, 10000)

        pm_peak = mass_peak(log_masses, L_pk, e_pk)
        pm_e = mass_energy(log_masses, Eg, e_eg)
        pm_P = mass_plat(log_masses, P, e_P)

        pm = pm_P * pm_e 

        pm /= np.sum(pm * (log_masses[1] - log_masses[0]))

        if do_hills:
            try:
                import tidalspin as ts
            except ImportError  as e:
                print(e)
                warn("tidalspin is not avaliable, please install it.  (See README)", stacklevel=4)

            prior_m = lambda logM: pm[np.argmin(abs(logM-log_masses))]

            phm, logm = ts.mass_posterior(prior_Mbh=prior_m, 
                                        log_Mbh_min=min(log_masses), 
                                        log_Mbh_max=max(log_masses), 
                                        N_bh = 500)
            
            pm_peak = mass_peak(logm, L_pk, e_pk)
            pm_e = mass_energy(logm, Eg, e_eg)
            pm_P = mass_plat(logm, P, e_P)

            pm = pm_P * pm_e 

            pm /= np.sum(pm * (logm[1] - logm[0]))


            return logm, pm_peak, pm_e, pm_P, pm, phm
        return log_masses, pm_peak, pm_e, pm_P, pm



