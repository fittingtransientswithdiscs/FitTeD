# FitTeD

[![PyPI](https://img.shields.io/pypi/v/astro-fitted.svg)](https://pypi.org/project/astro-fitted/)
[![Python](https://img.shields.io/pypi/pyversions/astro-fitted.svg)](https://pypi.org/project/astro-fitted/)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://fittingtransientswithdiscs.github.io/FitTeD/)

**Fit**ting **T**transi**E**nts with **D**iscs — a public Python package for fitting the
light curves and spectra of astrophysical transients with evolving relativistic accretion
discs.

Hey. Thanks for using FitTeD.

The paper describing this package is
[Mummery, Nathan, Ingram & Gardner (2025), MNRAS 544, 2225](https://ui.adsabs.harvard.edu/abs/2025MNRAS.544.2225M/abstract).
Please cite it if you make use of this code in your research — there is a BibTeX entry in
`cite_fitted.bib`.

## Install

```bash
pip install astro-fitted
```

That is the whole installation. FitTeD is pure Python: no compiler, no build step, and the
same install works on Linux, macOS and Windows.

The distribution is called `astro-fitted` because plain `fitted` on PyPI belongs to an
unrelated package. The import name is just `fitted`:

```python
import fitted
```

To work from a checkout instead, which you want if you plan to modify the code:

```bash
git clone https://github.com/fittingtransientswithdiscs/FitTeD.git
cd FitTeD
python3 -m pip install -e .
```

### Optional: manyTDE

If you want to load TDE data sets by IAU name rather than supplying your own photometry,
you will also need [manyTDE](https://github.com/sjoertvv/manyTDE), Andy and Sjoert's
database of optical/UV TDE light curves. Nothing else needs it.

```bash
git clone https://github.com/sjoertvv/manyTDE.git
cd manyTDE && python3 -m pip install -e .
```

## Documentation

**https://fittingtransientswithdiscs.github.io/FitTeD/**

Eight tutorials, all executed with their outputs, taking you from loading a light curve to
a converged MCMC fit of AT2019dsg. If you are new to the package, start there rather than
here — tutorial 00 has you computing a model in about a minute, and tutorial 06 is a
complete analysis end to end.

## Running the examples

There are four example scripts in `fitted/examples/`, which are the older and terser route
through the same material.

`data_loading.py` sets up a FitTeD `Data_Set` for the tidal disruption event AT2019dsg and
does some processing (see the code and paper for details).

`fitting_models.py` shows how to generate FitTeD models and fit them to data in various
ways. Switch `yes_i_want_to_find_a_best_fit` to `True` for a best fit (about a minute), or
`yes_i_want_to_run_a_chain` to `True` for a proper analysis — though the chain takes around
five hours with the current settings.

`analysis.py` then shows some of the science you can get out of the fit.

## Version 2.0.0

The Fortran backend is gone; the package is pure Python throughout. If you have fitted with
FitTeD before, read [CHANGELOG.md](CHANGELOG.md) — in particular the note on the radial
grid default, which changes your numbers slightly.

---

Happy TDE-ing.

Cheers,
Andy\*, Ed, and Adam

P.S. any comments or questions, drop me a line.

\* amummery@ias.edu
