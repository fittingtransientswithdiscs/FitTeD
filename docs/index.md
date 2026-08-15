# FitTeD

**Fit**ting **T**ransi**e**nts with **D**iscs — a public Python package for fitting light
curves and spectra of astronomical transients with evolving relativistic accretion discs.

FitTeD solves the time-dependent general relativistic disc equations, turns the resulting
temperature profile into multi-band light curves and spectra, and fits them to your data.
All of the relevant relativistic optics is included: Doppler shifting, gravitational
energy shifting, and gravitational lensing.

[Get started :material-arrow-right:](installation.md){ .md-button .md-button--primary }
[Tutorials](tutorials/README.md){ .md-button }

---

## What it does

<div class="grid cards" markdown>

-   **Time-dependent disc evolution**

    Self-consistent solutions of the relativistic disc equations, not a steady-state
    approximation. The disc spreads, the inner edge fills in, and the whole profile cools
    as mass drains through the ISCO.

-   **Optical, UV and X-ray together**

    The optical sits on the Rayleigh–Jeans tail and declines gently; the X-ray sits on the
    Wien tail and is exponentially sensitive to the inner-disc temperature. Fitting both
    is what constrains the black hole mass.

-   **Relativistic ray tracing**

    A pure-Python Kerr ray tracer, so photon transport needs no compiler and no build
    step on any platform.

-   **MCMC and posteriors**

    emcee, parallelised across walkers, with convergence diagnostics, corner plots and
    posterior light curves built in.

</div>

## What it is used for

- **Tidal disruption events** — the late-time optical/UV and X-ray emission
- **X-ray binaries** — soft-state light curve decays
- **Luminous fast blue optical transients** — accretion-powered transients

## Thirty seconds of code

```python
import fitted

# Load a TDE by name, from the manyTDE catalogue
data = fitted.data.Data_Set(
    manyTDE_name='AT2019dsg',
    manyTDE_bands=['r.ztf', 'g.ztf', 'UVW1.uvot'],
    global_systematic=0.1,
)

# Build a disc model with an early-time component
model = fitted.models.GR_disc(data=data, decay=True, decay_type='pl', rise=True)
fit = fitted.Fit(model=model)

# Physical values in, sampling-space vector out
start = model.pack_parameters(
    log_mh=6.7, a_bh=0.01, m_disc=0.05, r0=30.0, tvi=15.0, t0=-2.0, incl=70.0,
    log_L=43.3, t_fb=67.0, p=5/3, t_peak=1.0, sigma=8.0, log_T=4.4,
)

best = fit.best_fit(start)
fit.run_chain(full_pars=best, nwalkers=100, nsteps=2000, backend_path='chain.h5')
fit.plot_corner()
```

[Tutorial 06](tutorials/06_complete_workflow_at2019dsg.ipynb) is this, done properly, end
to end, with real output.

## Where to start

If you are new to the package, read the tutorials in this order: **00** (install),
**01** (data), **03** (model), **04** (fitting), **05** (MCMC). That is a complete arc and
needs no optional dependencies.

!!! tip "Read tutorial 03 even if you are in a hurry"

    It explains the difference between the parameters the physics uses (`m_disc` in solar
    masses, `incl` in degrees) and the parameters the sampler moves (`log_m_disc`,
    `cos_incl`). Confusing the two is the most common way to end up with a chain that
    will not start, and `model.pack_parameters()` makes the problem go away entirely.

## Citation

Mummery et al. (2025), *Fitting transients with discs (FitTeD): a public light curve and
spectral fitting package based on evolving relativistic discs*, MNRAS **544**, 2225.

See [Citing FitTeD](citing.md) for BibTeX.
