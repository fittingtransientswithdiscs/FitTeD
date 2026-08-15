import fitted
import matplotlib.pyplot as plt
import numpy as np
from astropy import units

fitted.format_plots()

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
                         args_X=[[timeX, lumX, errX, [0.3, 10]]], bands_X=['Swift XRT'], 
                         args_X_upperlim=[[timeXUL, lumXUL, N_sigmaX, [0.3, 10]]], bands_X_upperlim=['Swift XRT upper limit'], 
                         global_systematic=0.1)#global_systematic takes error to error + global_systematic * luminosity. 

                        #### if you have raw data on file which is not in manyTDE you would use 
                        #### args_UV = [times, lums, errs, freq_of_band_in_Hz], bands_UV=[name_of_band]. 
                        #### to load it in. 
                        #### one can also add
                        #### args_UV_upperlim = [times, lums, Nsigma, freq_of_band_in_Hz], bands_UV_upperlim=[name_of_band]. 

### NOTE, I USE THE NAME 'UV' TO REFER TO ANY OBSERVATION THAT IS NOT ACROSS A BROAD BAND, AND 'X' FOR ANYTHING THAT IS. 
### I APPRECIATE THAT THIS IS POTENTIALLY CONFUSING BUT SADLY YOU WILL HAVE TO DEAL WITH IT. 


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

########################################################
## Example 2: Using observer-frame fluxes
########################################################

print("\n" + "="*70)
print("Example 2: Using Observer-Frame Fluxes")
print("="*70)

# Example: Custom observer-frame flux data
# Assume we have data for a source at z=0.1
z_custom = 0.1
# Distance will be automatically calculated from redshift using astropy.cosmology

# Observer-frame data
times_obs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])  # days since peak
fluxes_g = np.array([1.2e-15, 1.0e-15, 8e-16, 6e-16, 4e-16])  # erg/s/cm²/Hz
errors_g = fluxes_g * 0.1  # 10% errors
frequency_g = 6.0e14  # Hz (observer frame, ~500 nm)

fluxes_r = np.array([1.5e-15, 1.2e-15, 1.0e-15, 8e-16, 6e-16])  # erg/s/cm²/Hz
errors_r = fluxes_r * 0.1
frequency_r = 4.5e14  # Hz (observer frame, ~670 nm)

# Create data set with observer-frame fluxes
# Distance is automatically calculated from redshift
d_flux = fitted.data.Data_Set(
    args_UV_flux=[
        [times_obs, fluxes_g, errors_g, frequency_g],
        [times_obs, fluxes_r, errors_r, frequency_r]
    ],
    bands_UV_flux=['g.custom', 'r.custom'],
    redshift=z_custom
)

print(f"Created data set with {len(d_flux.bands_UV)} bands: {d_flux.bands_UV}")
print(f"Redshift: {d_flux.redshift}")
print(f"Distance: {d_flux.d_Mpc:.2f} Mpc")

# Plot the data
fig3 = d_flux.plot_data(ylim=(1e40, 2e44))
plt.title('Observer-Frame Flux Input (Converted to Rest-Frame)')
# plt.savefig('observer_frame_flux_example')
# plt.show()

########################################################
## Example 3: Using AB magnitudes
########################################################

print("\n" + "="*70)
print("Example 3: Using AB Magnitudes")
print("="*70)

# Example: Custom AB magnitude data
# Same source at z=0.1
times_obs_mag = np.array([0.0, 10.0, 20.0, 30.0])
mags_g = np.array([20.0, 20.2, 20.5, 20.8])  # AB magnitudes
err_mags_g = np.array([0.1, 0.1, 0.1, 0.1])
frequency_g_mag = 6.0e14  # Hz (observer frame)

mags_r = np.array([19.8, 20.0, 20.3, 20.6])  # AB magnitudes
err_mags_r = np.array([0.1, 0.1, 0.1, 0.1])
frequency_r_mag = 4.5e14  # Hz (observer frame)

# Create data set with AB magnitudes
d_mag = fitted.data.Data_Set(
    args_UV_mag=[
        [times_obs_mag, mags_g, err_mags_g, frequency_g_mag],
        [times_obs_mag, mags_r, err_mags_r, frequency_r_mag]
    ],
    bands_UV_mag=['g.mag', 'r.mag'],
    redshift=z_custom
)

print(f"Created data set with {len(d_mag.bands_UV)} bands: {d_mag.bands_UV}")
print(f"Redshift: {d_mag.redshift}")
print(f"Distance: {d_mag.d_Mpc:.2f} Mpc")

# Plot the data
fig4 = d_mag.plot_data(ylim=(1e40, 2e44))
plt.title('AB Magnitude Input (Converted to Rest-Frame)')
# plt.savefig('ab_magnitude_example')
# plt.show()

########################################################
## Example 4: Mixing manyTDE with custom flux data
########################################################

print("\n" + "="*70)
print("Example 4: Mixing manyTDE with Custom Flux Data")
print("="*70)

# Load some bands from manyTDE
bands_from_manyTDE = ['g.ztf', 'r.ztf']

# Add custom observer-frame flux data for an additional band
times_custom = np.array([0.0, 5.0, 10.0, 15.0])
fluxes_custom = np.array([1e-15, 9e-16, 8e-16, 7e-16])
errors_custom = fluxes_custom * 0.15
frequency_custom = 8.0e14  # Hz (observer frame, UV band)

try:
    d_mixed = fitted.data.Data_Set(
        manyTDE_name='AT2019dsg',
        manyTDE_bands=bands_from_manyTDE,
        args_UV_flux=[[times_custom, fluxes_custom, errors_custom, frequency_custom]],
        bands_UV_flux=['UV.custom'],
        redshift=0.1  # Required when using flux inputs
    )
    
    print(f"Created mixed data set with {len(d_mixed.bands_UV)} bands: {d_mixed.bands_UV}")
    print(f"  - manyTDE bands: {bands_from_manyTDE}")
    print(f"  - Custom flux band: UV.custom")
    
    # Plot all bands
    fig5 = d_mixed.plot_data(ylim=(1e40, 2e44))
    plt.title('Mixed: manyTDE + Custom Flux Data')
    # plt.savefig('mixed_data_example')
    # plt.show()
    
except Exception as e:
    print(f"Note: manyTDE not available, skipping mixed example: {e}")

print("\n" + "="*70)
print("Examples Complete")
print("="*70)
print("\nKey points:")
print("  - Observer-frame times are automatically converted: t_rest = t_obs / (1+z)")
print("  - Observer-frame frequencies are automatically converted: ν_rest = ν_obs * (1+z)")
print("  - Observer-frame fluxes are converted to rest-frame vL_v luminosities")
print("  - AB magnitudes are converted to fluxes, then to rest-frame vL_v")
print("  - All data sources (manyTDE, args_UV, flux, magnitude) can be mixed")
print("  - Band names from all sources are correctly appended to bands_UV")