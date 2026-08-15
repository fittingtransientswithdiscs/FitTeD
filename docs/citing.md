# Citing FitTeD

If you use FitTeD in your research, please cite the method paper.

## Paper

Mummery, A., Nathan, E., Ingram, A. & Gardner, M. (2025), *Fitting transients with discs
(FITTED): a public light curve and spectral fitting package based on evolving
relativistic discs*, **MNRAS 544, 2225–2240**, [doi:10.1093/mnras/staf1565](https://doi.org/10.1093/mnras/staf1565).

- [arXiv:2408.15048](https://arxiv.org/abs/2408.15048)
- [ADS](https://ui.adsabs.harvard.edu/abs/2025MNRAS.544.2225M/abstract)

## BibTeX

```bibtex
@ARTICLE{2025MNRAS.544.2225M,
       author = {{Mummery}, Andrew and {Nathan}, Edward and {Ingram}, Adam and {Gardner}, M.},
        title = "{Fitting transients with discs (FITTED): a public light curve and spectral fitting package based on evolving relativistic discs}",
      journal = {\mnras},
     keywords = {accretion, accretion discs, black hole physics, transients: tidal disruption events, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2025,
        month = dec,
       volume = {544},
       number = {2},
        pages = {2225-2240},
          doi = {10.1093/mnras/staf1565},
archivePrefix = {arXiv},
       eprint = {2408.15048},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.544.2225M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

The authoritative copy is `cite_fitted.bib` in the repository root. The block above is
checked against it on every documentation build, so the two cannot drift apart.

## Software version

If the specific version matters to your result — and for anything involving the disc
numerics it does — cite the release as well as the paper. Record the version you ran:

```python
import fitted
print(fitted.__version__)
```

## Components with their own provenance

**Ray tracing.** The Kerr ray tracer in `fitted/models/raytrace/` is a vendored subset of
`kerrgeo`, with a method paper in preparation. See `raytrace/PROVENANCE.md` in the
repository for exactly which version was vendored and what was excluded.

**Photometry.** If you loaded data with `manyTDE_name=`, the photometry comes from
[manyTDE](https://github.com/sjoertvv/manyTDE) (van Velzen et al.) and should be cited
accordingly, along with the original sources for the individual light curves.

**Cosmology.** Distances are computed with `astropy.cosmology.Planck18` unless you supply
your own.
