import numpy as np
import matplotlib.pyplot as plt 
import fitted 
fitted.format_plots()

tde = 'AT2019dsg'## Makes plots shown in Figure 3 of the paper. 
bands_i_want = ['B.uvot', 'U.uvot', 'g.ztf', 'UVW1.uvot', 'UVW2.uvot', 'UVM2.uvot']## Uses manyTDE data base

d = fitted.data.Data_Set(manyTDE_name=tde, manyTDE_bands=bands_i_want)

#### if you have raw data on file which is not in manyTDE you would use 
####
#  args_UV = [[times1, lums1, errs1, freq_of_band1_in_Hz], 
#             [times2, lums2, errs2, freq_of_band2_in_Hz], 
#               .... ]
#
#     bands_UV=[name_of_band1, name_of_band2, ...]. 
# 
#### Where, times_k (etc.) is a list of the times of the kth band.
### 
### Make sure that each data set is corrected to the REST FRAME OF THE SOURCE!  

tmin = -100
for band in d.bands:
    d.remove_before_time(band, t_start=tmin)## Delete pre peak data for simplicity

rise = False

m = fitted.models.TDEFLARE(data=d, rise=rise)## TDEFLARE model
f=fitted.FitTDEFLARE(model=m)## Fitting package for TDEFLARE. 

f.run_chain(init_key_pars=[41.5, 4.3], nwalkers=100, nsteps=1000, 
        backend_name='TDEFLARE_{}'.format(tde), 
        backend_path='TDEFLARE_{}'.format(tde), 
        overwrite_backend=True)## Runs MCMC chain. 

f.plot_corner()## Corner plot.

f.plot_walkers()## Walkers

fig = f.plot_data()## Plots light curves
times = np.linspace(0, 2000, 100)
f.plot_posterior_lightcurves(times, fig=fig)## And model. 

logm, pm_peak, pm_e, pm_P, pm_conflate, pm_conflate_hills = f.get_mass_constraints(do_hills=True)## Does Hills mass integral
## Requires tidalspin code to do the Hills mass integral 
## See : https://github.com/andymummeryastro/tidalspin

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(logm, pm_peak, 'g', label=r'$p(\log_{10}M_\bullet|L_{\rm pk})$')## Individual
ax.plot(logm, pm_e, 'purple', label=r'$p(\log_{10}M_\bullet|E_{g})$')## Mass
ax.plot(logm, pm_P, 'blue', label=r'$p(\log_{10}M_\bullet|L_{P})$')## Posteriors


ax.plot(logm, pm_conflate, 'k', label=r'$p(\log_{10}M_\bullet|E_g, L_P)$')## Conflation
ax.plot(logm, pm_conflate_hills, 'r', label=r'$p(\log_{10}M_\bullet)$')## Hills integral

ax.set(xlim=(5, 8.5), 
       ylabel='Probability density function', 
       xlabel=r'$\log_{10}M_\bullet/M_\odot$')

ax.legend(edgecolor='w')

plt.show()