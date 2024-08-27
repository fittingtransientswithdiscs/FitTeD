from astropy import constants as con
from astropy import units as unitss
from numpy import pi 

G = con.G.value
Ms = con.M_sun.value
c = con.c.value
h = con.h.value
kb = con.k_B.value
O_sb = con.sigma_sb.value
keV_to_Hz =  1000 * con.e.value / h # 2.42e17 Hz
r_g = G*Ms/(c*c)

kelvin_to_keV = (unitss.K * con.k_B).to(unitss.keV).value
keV_to_erg = unitss.keV.to(unitss.erg)

H0 = 67.4#km s^{-1} Mpc^{-1} #hubble's constant
omega_m = 0.315 #the ratio of the density of the Universe to the critical density
omega_l = 0.685 #cosmological constant, normalized to critical desnsity
