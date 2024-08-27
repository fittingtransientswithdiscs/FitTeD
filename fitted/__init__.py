from . import models, prior, data, constants
from .fit import *
__version__ = '1.0.0' 

def print_starting_message():
    print("  _____  _  _   _____      ____   ", end="\n")
    print(" |  ___|(_)| |_|_   _|___ |  _ \  ", end="\n")
    print(" | |_   | || __| | | / _ \| | | | ", end="\n")
    print(" |  _|  | || |_  | ||  __/| |_| | ", end="\n")
    print(" |_|    |_| \__| |_| \___||____/  ", end="\n")
    print("                                  ", end="\n")
    print("   FITing TransiEnts with Discs   ", end="\n")
    print("")
    print(" Version", __version__)
    print("")
    print(" GR Photon treatment avaliable?  ", models.numerical_disc.numerical_disc_avaliable)
    print(" manyTDE compatible?  ", data.manyTDE_available)
    print("")
    print(" FitTeD was created by Andrew Mummery* and Edward Nathan$.", end="\n")
    print(" It is still in development. ",end="\n")
    print("")
    print(" * andrew.mummery@physics.ox.ac.uk",end="\n")
    print(" $ enathan@caltech.edu",end="\n")
    print("")


print_starting_message()