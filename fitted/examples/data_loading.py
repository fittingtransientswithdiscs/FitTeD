import fitted
import matplotlib.pyplot as plt
import numpy as np
from astropy import units

D_19dsg=236.4# Mpc
data = np.genfromtxt('2019dsg_Xray.txt', delimiter='', skip_header=3)## Data from Stein + 2021.
bands_d = np.genfromtxt('2019dsg_Xray.txt', delimiter='', skip_header=3, usecols=(1), dtype=str)## Data from Stein + 2021.

timeX = data[:, 0][bands_d=='X']
lumX = data[:, 2][bands_d=='X'] * 4 * np.pi * (D_19dsg * units.Mpc.to(units.cm))**2.0 
errX = data[:, 3][bands_d=='X'] * 4 * np.pi * (D_19dsg * units.Mpc.to(units.cm))**2.0 

timeXUL = data[:, 0][bands_d=='XUL']
lumXUL = data[:, 2][bands_d=='XUL'] * 4 * np.pi * (D_19dsg * units.Mpc.to(units.cm))**2.0 
N_sigmaX = 3 * np.ones_like(lumXUL)## Significance of upper limits

bands_i_want = ['r.ztf', 'g.ztf', 'UVW1.uvot', 'UVW2.uvot', 'UVM2.uvot']
d = fitted.data.Data_Set(manyTDE_name='AT2019dsg', manyTDE_bands=bands_i_want, 
                         args_X=[timeX, lumX, errX, [0.3, 10]], bands_X=['Swift XRT'], 
                         args_X_upperlim=[timeXUL, lumXUL, N_sigmaX, [0.3, 10]], bands_X_upperlim=['Swift XRT upper limit'], 
                         global_systematic=0.1)#global_systematic takes error to error + global_systematic * luminosity. 

                        #### if you have raw data on file which is not in manyTDE you would use 
                        #### args_UV = [times, lums, errs, freq_of_band_in_Hz], bands_UV=[name_of_band]. 
                        #### to load it in. 
                        #### one can also add
                        #### args_UV_upperlim = [times, lums, Nsigma, freq_of_band_in_Hz], bands_UV_upperlim=[name_of_band]. 

### NOTE, I USE THE NAME 'UV' TO REFER TO ANY OBSERVATION THAT IS NOT ACROSS A BROAD BAND, AND 'X' FOR ANYTHING THAT IS. 
### I APPRECIATE THAT THIS IS POTENTIALLY CONFUSING BUT SADLY YOU WILL HAVE TO DEAL WITH IT. 

d.format_plots()

d.bands_systematic['r.ztf'] = 0.3## r-band is noisy and messes with the fits, this makes the algorithms ignore it more, but not completely. 

fig1 = d.plot_data(ylim=(1e40, 2e44))

for band in bands_i_want:
    d.remove_before_time(band, t_start=-100)## Some data cleaning of ZTF, etc. 
    d.rebin_band(band, delta_t=10, t_end=100)
    d.rebin_band(band, delta_t=50, t_start=100)


d.save('AT2019dsg_data_processed')# to avoid having to do the above every time. 

fig2 = d.plot_data(ylim=(1e40, 2e44))### Notation obvious

# fig1.savefig('all_data_raw')
# fig2.savefig('all_data_binned')

plt.show()