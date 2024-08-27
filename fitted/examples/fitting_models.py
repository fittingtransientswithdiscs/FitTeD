import fitted
import matplotlib.pyplot as plt
import numpy as np

yes_i_want_to_run_a_chain = False## naming convention obvious -- takes ~ 5 hours
save_str = 'GR_disc_with_rise_pl_decay'
nwalk=100
nstep=3000
yes_i_want_to_find_a_best_fit = False## naming convention obvious -- takes ~ 1 minute.

# decay_type = 'exp'#exponential decay for early time
decay_type = 'pl'#power law decay for early time
rise = True# if True will fit a Gaussian rise and peak timescale. 

d = fitted.data.Data_Set()
d.load('AT2019dsg_data_processed')
m = fitted.models.GR_disc(data=d, decay_type=decay_type, rise=rise)## GR_disc significantly faster than GR_disc_plus (at cost of neglecting photon physics).
f = fitted.Fit(model=m)
f.format_plots()

d.plot_data(ylim=(1e40, 2e44))## if you want only a few bands do bands=[...]


"""
Parameters of the disc/photon model are:

log_mh -- log_10 black hole mass (units = log_10 solar masses)
a_bh -- black hole spin (-1 < a_bh < 1), (units = dimensionless)
m_disc -- disc mass (units = solar masses)
r0 -- initial disc ring radius (units = gravitational radii)
tvi -- `viscous` timescale of disc (really just evolutionary timescale), (units = days)
t0 -- time before peak of initial condition (this allows for the fact that our initial condition is over simplified), (units = days)
incl -- disc-observer inclination angle (units = degrees). More important for RelDisc. 
"""
### Not a fit -- just eyeballed. 
log_mh = 7.0
a_bh = 0.01
m_disc = 0.05
r0 = 30
tvi = 15
t0 = -2#positive = before peak, negative = after peak. 
incl = 70

"""
Decay model is either a power law or exponential -- you choose. 

Parameters of the decay model are:

log_L -- single temperature blackbody amplitude (units = log_10 erg/s)
tdecay -- either e-folding time (exp model) or fall-back time (pl model), units = days
p -- power-law index (pl model ONLY), dimensionless
log_T -- single temperature blackbody temperature (units = log_10 Kelvin)
"""
log_L = 43.3
tdecay = 67
log_T = 4.8
p = 5/3

"""
Rise model is a Gaussian rise

Parameters of the rise model are:
sigma -- rise timescale (units = days)
t_peak -- time of light curve peak (units = days)
"""
sigma = 8
t_peak = 1


if not rise:#### The following code shows how to fit all the different early time model options. Delete as necesary. 
    if decay_type == 'exp':
        pars = [log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay,  log_T]
        log_L, tdecay, log_T = f.get_early_model_pars(p0=[log_L, tdecay, log_T], t_cut=200)## rise and decay model fit, fixed discs
        if yes_i_want_to_find_a_best_fit:
            log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, log_T = f.best_fit(init_guess=pars, print_best_fit=True)## fit all pars
            pars = [log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, log_T]

    elif decay_type == 'pl':
        pars = [log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, p, log_T]
        log_L, tdecay, p, log_T = f.get_early_model_pars(p0=[log_L, tdecay, p, log_T], t_cut=200)## rise and decay model fit, fixed discs
        if yes_i_want_to_find_a_best_fit:
            log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, p, log_T = f.best_fit(init_guess=pars, print_best_fit=True)## fit all pars
            pars = [log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, p, log_T]
else:
    if decay_type == 'exp':
        pars = [log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, t_peak, sigma, log_T]
        log_L, tdecay, t_peak, sigma, log_T = f.get_early_model_pars(p0=[log_L, tdecay, t_peak, sigma, log_T], t_cut=200)## rise and decay model fit, fixed discs
        if yes_i_want_to_find_a_best_fit:
            log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, t_peak, sigma, log_T = f.best_fit(init_guess=pars, print_best_fit=True)## fit all pars
            pars = [log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, t_peak, sigma, log_T]

    elif decay_type == 'pl':
        pars = [log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, p, t_peak, sigma, log_T]
        log_L, tdecay, p, t_peak, sigma, log_T = f.get_early_model_pars(p0=[log_L, tdecay, p, t_peak, sigma, log_T], t_cut=200)## rise and decay model fit, fixed discs
        if yes_i_want_to_find_a_best_fit:
            log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, p, t_peak, sigma, log_T = f.best_fit(init_guess=pars, print_best_fit=True)## fit all pars
            pars = [log_mh, a_bh, m_disc, r0, tvi, t0, incl, log_L, tdecay, p, t_peak, sigma, log_T]


tplot = np.linspace(-100, 1600, 1500)
tplot = np.linspace(10, 1600, 150)

for band in d.bands_UV:### could change bands if wanted
    lmod = m.model_UV(tplot, log_mh, a_bh, m_disc, r0, tvi, t0, incl, v=d.bands_freq[band])
    if not rise:#### all the different ways to plot models
        if decay_type == 'exp':
            emod = m.decay_model(tplot, log_L, tdecay, log_T, v=d.bands_freq[band])
        elif decay_type == 'pl':
            emod = m.decay_model(tplot, log_L, tdecay, p, log_T, v=d.bands_freq[band])
    else:
        if decay_type == 'exp':
            emod = m.decay_model(tplot, log_L, tdecay, t_peak, log_T, v=d.bands_freq[band]) + m.rise_model(tplot, log_L, sigma, t_peak, log_T, v=d.bands_freq[band])
        elif decay_type == 'pl':
            emod = m.decay_model(tplot, log_L, tdecay, p, t_peak, log_T, v=d.bands_freq[band]) + m.rise_model(tplot, log_L, sigma, t_peak, log_T, v=d.bands_freq[band])

    plt.plot(tplot, lmod, '--', c=d.band_colours[band])###disc model
    plt.plot(tplot, lmod+emod, c=d.band_colours[band])###total model
    plt.plot(tplot, emod, '-.', c=d.band_colours[band])###early model

lmod = m.model_X(tplot, log_mh, a_bh, m_disc, r0, tvi, t0, incl, El=d.bands_freq['Swift XRT'][0], Eh=d.bands_freq['Swift XRT'][1])
plt.plot(tplot, lmod, '--', c=d.band_colours['Swift XRT'])


if yes_i_want_to_run_a_chain:    
    f.run_chain(full_pars=pars, nwalkers=nwalk, nsteps=nstep, scatter=0.001, backend_path='AT2019dsg_backend_'+save_str)## mcmc all pars
    f.save('AT2019dsg_'+save_str)


plt.show()