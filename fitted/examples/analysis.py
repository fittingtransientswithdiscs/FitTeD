import numpy as np
import fitted
import matplotlib.pyplot as plt 

fitted.format_plots()

f = fitted.Fit()
f.load('AT2019dsg_GR_disc_with_rise_pl_decay')

if 0:
    tplot = np.geomspace(1, 1750, 100) 
    fig = f.plot_data(ylim=(1e40, 2e44), bands=['UVW1.uvot', 'g.ztf', 'Swift XRT', 'Swift XRT upper limit'])
    fig = f.plot_posterior_lightcurves(N=500, bands=['UVW1.uvot', 'g.ztf', 'Swift XRT upper limit'], fig=fig, t_plot=tplot, 
                                    as_range=True, plotted_range=[5, 95], plot_median=True)

if 0:
    todayMJD = 60603.87##~20th October 2024. 
    tmax = todayMJD - 58603.87
    time = np.array([100, tmax])
    fig = f.plot_density_posterior(t_plot=time, N=500, colors=['b', 'g'], in_cgs=True, plotted_range=[5, 95])
    fig2 = f.plot_density_posterior(t_plot=time, N=500, colors=['b', 'g'], in_cgs=False, plotted_range=[5, 95])


if 0:
    tplot = np.geomspace(1, 2000, 200)
    fig = f.plot_bolometric_luminosity_posterior(t_plot=tplot, N=200, as_eddington=True)

if 0:
    fig = f.plot_walkers(just_disc=True)

if 0:
    fig = f.plot_corner(just_disc=True, f_discard=0.8, color='g')
    axes = fig.get_axes()

    for m in [0, 7, 14, 21, 28, 35, 42]:
        axes[m].axvline(6.7, ls='-',c='k')
        axes[m].axvline(6.3, ls='--',c='k')
        axes[m].axvline(7.1, ls='--',c='k')
        axes[m].set_xlim(6.1)
    axes[2].plot(0, 0, c='k', label=r'$M_\bullet(\sigma_\star)$')
    axes[2].legend(framealpha=0, edgecolor='w', fontsize=25)


plt.show()