# README #

Hey. 

Thanks for using FitTeD. 

The paper describing this package is here: https://ui.adsabs.harvard.edu/abs/2025MNRAS.544.2225M/abstract
Please cite if you make use of this code in your research.

### Setup ###

To get this all up and running, you will need to run the following:

* Pull into it's own folder
* Within the folder, run "python3 -m pip install -e ." 
* The -e flag is required so that the package is editable, and therefore the following steps install the fortran code correctly.
* Go into the numerical_disc folder (fitted/models/numerical_disc/) 
* Compile with the "make" command
* If it cannot find a fortran compiler, set the FC environment variable.  e.g., "make FC='gfortran-mp-9'" 

To run the below examples you will need to also have manyTDE installed (Andy and Sjoert's database of optical/UV TDE light curves). 

* This is only necessary if you want to get TDE data sets by IAU name, rather than loading your own data. 
* manyTDE can be found here: https://github.com/sjoertvv/manyTDE
* Again just run “ python3 -m pip install -e . ” in the manyTDE directory to set up. 

### Running the code ###

There are 3 example scripts in the fitted/examples directory which run through how this all works. 

Running 
> data_loading.py 

Will set up a FitTeD Data_Set class for the tidal disruption event AT2019dsg and do some processing (see code/paper for details).

Running 
> fitting_models.py

will show you how to generate FitTeD models, and fit them to data in various ways. 

Switch the variables `yes_i_want_to_run_a_chain` and `yes_i_want_to_find_a_best_fit` to True if you want to do a proper analysis, although running the chain will take ~ 5 hours with current settings (on my laptop). 

If you do run the chain, then 
> analysis.py

shows you some science results you can get from the fit. 

Happy TDE-ing. 

Cheers,
Andy*, Ed, and Adam

P.S., any comments/questions, drop me a line. 

* amummery@ias.edu
