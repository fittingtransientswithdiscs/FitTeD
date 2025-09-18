from . import models, prior, data, constants
from .fit import *
__version__ = '1.0.3' 

_message=r"""
  _____  _  _   _____      ____   
 |  ___|(_)| |_|_   _|___ |  _ \  
 | |_   | || __| | | / _ \| | | | 
 |  _|  | || |_  | ||  __/| |_| | 
 |_|    |_| \__| |_| \___||____/  
                                  
   FITing TransiEnts with Discs   

 Version {version:s}

 GR Photon treatment avaliable?  {numdiscavaliable:s}
 manyTDE compatible?  {manytdeavaliable:s}

 FitTeD was created by Andrew Mummery* and Edward Nathan$.

 Please cite us if you make use of this code
 https://ui.adsabs.harvard.edu/abs/2024arXiv240815048M/abstract

 * amummery@ias.edu"
 $ enathan@caltech.edu"

"""

def print_starting_message():
    print(_message.format(version=__version__,
                         numdiscavaliable=models.numerical_disc.numerical_disc_avaliable,
                         manytdeavaliable=data.manyTDE_available
                        ), 
          flush=True)


def format_plots():
    """
        Formats all plots as default.  
    """
    import os
    import matplotlib.pyplot as plt
    fullpath = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(fullpath+'/fitted_style.mplstyle')

print_starting_message()