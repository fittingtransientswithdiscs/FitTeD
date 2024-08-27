from warnings import warn
import numpy as np
from astropy import constants, units
from functools import wraps

__all__ = ["reldisc_model", "N", "numerical_disc_avaliable", "energy_grid", "energy_grid_midpoints"]
tdedisc_grid_avaliable = False
numerical_disc_avaliable = "No"
try:
    from . import tdedisc_grid
except ImportError  as e:
    print(e)
    warn("tdedisc_grid is not avaliable, please compile it.  (See README)", stacklevel=4)
else:
    tdedisc_grid_avaliable = True
    numerical_disc_avaliable = "Yes"
    # This is the energy_grid we use.
    tdedisc_grid.setup_energy_grids()
    energy_grid = np.array( tdedisc_grid.internal_grids.earx)
    energy_grid_midpoints = np.array( tdedisc_grid.internal_grids.dEarr)


normalisation_unit = (units.keV * units.M_sun**-2 * units.s**-1 )#* (units.Mpc/units.cm)**2 )
r_g = constants.G / constants.c**2
N = ( r_g**2 * constants.sigma_sb * constants.k_B**-4 * units.keV**4 ).to(normalisation_unit)


def numerical_disc_decorator(func):
    '''
    Returns a decorator which checks to see if tdedisc_grid and dolfin are avaliable before attempting to call 
    numerical disc model
    '''
    @wraps(func)
    def numerical_disc_checker(*args, **kwargs):
        if not tdedisc_grid_avaliable:
            raise NotImplementedError("A compiled version of tdedisc_grid cannot be found")
        return func(*args, **kwargs)
    return numerical_disc_checker

@numerical_disc_decorator
def numerical_disc_model(bh_a, rout, incl, kTdisc_array, rdisc_array):
    '''
    Returns in (keV (Mpc/cm)^2 s^{−1} K^{−4}), so will need to be scaled to ergs to be compatable with the rest of fitreldisc
    Also, still needs to be scaled by ( ( M/M_{solar}) / (D / Mpc) )^2
    
    Look at N.unit to see exact units returned
    
    times must be interable, not a single number.
    
    The output of the fortran code is photar (photon array), which this wrapper also multiplies by N, an avaliable constant
    within this module.
    '''
    
    # Get temperature profile
    
    
    # taus = (np.array(times)+t0)/tvi## Will be passed at each time.
    # Can't use reldisc_at_times as it has an unknown memory bug
    return N.value * np.array( [tdedisc_grid.tdedisc_grid([bh_a, rout, incl], ri=rdisc_array,kti= kTdisc_array, nr=len(rdisc_array))] )
  
#  param(1) = 0.998    ! BH spin parameter
#  param(2) = 1e4    ! rout
#  param(3) = 70.0    ! Inclination angle in degrees
 
if __name__ == "__main__":
    print(numerical_disc_avaliable)
