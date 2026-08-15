# FitTeD tutorials

Eight notebooks. Read 00–06 in order; 07 is a reference to dip into. Every one ships with
its outputs already computed, so you can read them all without running anything — and every
code cell is real, so you can re-run them and get the same figures.

## The tutorials

| # | Tutorial | What it covers | Runs in | Needs manyTDE |
|---|----------|----------------|---------|---------------|
| 00 | [Introduction and installation](00_introduction_and_installation.ipynb) | What FitTeD does, how to install it, a first light curve | seconds | no |
| 01 | [Data loading basics](01_data_loading_basics.ipynb) | `Data_Set`, manyTDE, your own arrays, cutting and rebinning | seconds | optional |
| 02 | [Observer-frame data input](02_observer_frame_data_input.ipynb) | Fluxes, AB magnitudes, X-ray fluxes, and the cosmology FitTeD does for you | seconds | no |
| 03 | [Model setup and parameters](03_model_setup_and_parameters.ipynb) | The disc parameters, and physical vs sampled variables | seconds | no |
| 04 | [Basic fitting workflow](04_basic_fitting_workflow.ipynb) | Priors, `best_fit()`, and why which data you include matters more than the optimiser | ~1 min | no |
| 05 | [MCMC fitting with emcee](05_mcmc_fitting_with_emcee.ipynb) | Running a chain, sizing it, checking convergence, saving and resuming | ~5 min | no |
| 06 | [Complete workflow: AT2019dsg](06_complete_workflow_at2019dsg.ipynb) | All six bands, end to end, on a real TDE | ~35 min | **yes** |
| 07 | [Model options reference](07_model_options_reference.ipynb) | Every accuracy and performance option, and when to change it | ~3 min | optional |

Timings are for a laptop with a handful of cores; the MCMC notebooks parallelise across
walkers, so more cores is faster.

## Suggested path

**If you are new to the package**, read 00, 01, 03, 04, 05 in that order. That is the
complete arc — install, data, model, fit, posterior — and none of it needs optional
dependencies. Then 06 for the full multi-band analysis, which does need manyTDE.

!!! warning "Do not skip 03"

    It explains the difference between the parameters the physics uses (`m_disc` in solar
    masses, `incl` in degrees) and the parameters the sampler moves (`log_m_disc`,
    `cos_incl`). Getting these confused is by far the most common way to end up with a
    chain that will not start, and the fix — `model.pack_parameters()` — takes one line
    once you know about it.

**If you have your own photometry**, add 02.

**Before you spend a week of CPU time**, skim 07. One of the options in it can silently
ruin a long-baseline fit.

## A note on what these show

Tutorials 04, 05 and 06 fit real data and report what they actually get, including where
that is unflattering: chains that have not converged are reported as unconverged, and
notebook 04 shows a fit landing an order of magnitude from the right answer before showing
what fixes it.

Notebook 06 is explicit about two data-preparation traps that bias results — cutting
late-time data, and dropping negative fluxes from host-subtracted photometry. Both are easy
to do by accident, and both push the answer the same way.

## Running them

```bash
cd tutorials
jupyter lab            # or: jupyter notebook
```

The notebooks add the parent directory to `sys.path`, so they work from a bare checkout as
well as from an installed copy. If FitTeD is installed (`pip install -e .` in the repository
root) the installed copy takes precedence.

To re-execute the whole set from the command line, use the script rather than doing it by
hand — it runs them in order and then checks that every published notebook came out with its
outputs intact:

```bash
../tools/execute_tutorials.sh
```

## Drafts

`drafts/` holds notebooks that are **not** part of the published set and have never been
executed. See `drafts/README.md`. They are excluded from the website, and the docs build
fails if any of them is ever published by accident.

## Requirements

- Python 3.10 or newer
- FitTeD and its dependencies (see Tutorial 00)
- Jupyter
- **manyTDE** for Tutorial 06:
  `git clone https://github.com/sjoertvv/manyTDE && cd manyTDE && pip install -e .`

## Getting help

- `model.what_pars()` prints what any model instance expects, in order — start there when a
  parameter vector is rejected.
- The paper has the physics: [arXiv:2408.15048](https://arxiv.org/pdf/2408.15048)
- Example scripts: `fitted/examples/`

## Citation

Mummery et al. (2025), *Fitting transients with discs (FitTeD): a public light curve and
spectral fitting package based on evolving relativistic discs*, MNRAS **544**, 2225.
[arXiv:2408.15048](https://arxiv.org/abs/2408.15048) ·
[ADS](https://ui.adsabs.harvard.edu/abs/2025MNRAS.544.2225M/abstract)

BibTeX is in `cite_fitted.bib` in the repository root.
