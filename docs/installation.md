# Installation

## Quick version

```bash
git clone https://github.com/fittingtransientswithdiscs/FitTeD.git
cd FitTeD
python3 -m pip install -e .
```

That is enough to run everything in tutorials 00–07. There are no compiled extensions,
no compiler, and no build step: FitTeD is pure Python.

!!! note "Why `-e`?"

    The editable install lets you edit the package in place and see the change without
    reinstalling. It is not required — a plain `pip install .` works identically — but it
    is the configuration everything is tested in.

## Requirements

FitTeD is developed and tested on **Python 3.10 and newer**.

These are installed automatically by `pip`:

| Package | Used for |
|---------|----------|
| `numpy` | everything |
| `scipy` | special functions, optimisation |
| `astropy` | cosmology, units, constants |
| `matplotlib` | plotting |
| `emcee` | MCMC sampling |
| `corner` | corner plots |
| `h5py` | HDF5 chain backends |
| `numba` | the compiled Bessel-function kernels |
| `pandas`, `tqdm`, `importlib_resources` | data handling and progress bars |

## Optional: manyTDE

To load published TDE photometry by IAU name — `Data_Set(manyTDE_name='AT2019dsg', ...)` —
install [manyTDE](https://github.com/sjoertvv/manyTDE):

```bash
git clone https://github.com/sjoertvv/manyTDE.git
cd manyTDE
python3 -m pip install -e .
```

Only [tutorial 06](tutorials/06_complete_workflow_at2019dsg.ipynb) requires it. Everything
else uses data that ships with FitTeD or that you supply yourself.

## Checking it worked

```python
import fitted

print(fitted.__version__)
print("manyTDE available: ", fitted.data.manyTDE_available)
```

FitTeD prints a banner on import that reports the same thing. `manyTDE available: No`
is a perfectly good installation — only tutorial 06 needs it.

[Tutorial 00](tutorials/00_introduction_and_installation.ipynb) runs this check and then
computes a light curve, so if that notebook executes you are done.

## Running the tutorials

```bash
cd tutorials
jupyter lab
```

The notebooks add the parent directory to `sys.path`, so they work from a bare checkout as
well as an installed copy. If FitTeD is installed, the installed copy takes precedence.

Every notebook on this site ships with its outputs already computed, so you can read them
all without running anything.
