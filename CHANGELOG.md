# Changelog

## v2.0.0

The Fortran backend is gone and FitTeD is pure Python.

### Note for existing users: the default radial grid

The default `radial_grid_spacing` changed from linear to geometric, which
better resolves the inner disc. Evaluated at *fixed* parameters the model
output differs, most visibly in the X-ray, because the Wien tail depends
exponentially on the inner-disc temperature.

That same exponential sensitivity means the effect on a *fit* is small: the
inferred physical parameters shift only slightly to compensate. You should not
expect your conclusions to change, but you will not reproduce an old number
exactly without asking for the old configuration:

```python
model = GR_disc(..., radial_grid_spacing='linear', use_iv_approximation=False)
```

With those two options set, v2.0.0 reproduces v1.0.5 bit for bit — verified
over 60 random disc systems at four epochs and 24 frequencies, plus
band-integrated `model_UV` and `model_X`, with a maximum relative difference of
exactly zero.

### Removed

- **`GR_disc_plus_fortran`** — use `GR_disc_plus`, which is pure Python and
  faster. Agreement is sub-percent through the optical and soft X-ray; it
  widens far down the Wien tail, where the flux is many decades below peak.
- **The Fortran `numerical_disc` backend**, its Makefiles, `meson.build` and
  `BUILD_FORTRAN.md`. All of it is preserved at tag **`v1.0.5-final-fortran`**
  if you ever need it.

### Installation

- **No compiler, no meson, no ninja, no f2py.** One universal `py3-none-any`
  wheel that installs on Linux, macOS and Windows.
- **Python 3.10 or newer** (was 3.12, a meson-python constraint rather than a
  scientific one).
- The import banner no longer claims "GR Photon treatment avaliable? No" on
  machines without gfortran. That line was misleading for every user who never
  built the Fortran: the Python ray tracer was working the whole time.

### Fixed

- Packaging listed only the top-level package, so a setuptools install would
  have shipped without `models/` and without the bundled data files. Fixed with
  `find_packages` and explicit `package_data`.
- `numba` is now a declared dependency. `GR_disc` defaults to
  `use_iv_approximation=True`, so without numba the code silently fell back to
  `scipy.special.iv` and the published tutorial numbers were not reproducible.
- The documentation workflow only rebuilt on changes under `docs/`. Because
  `docs/tutorials` is a symlink, editing a tutorial never triggered a rebuild.
- `sync_to_public.sh` followed symlinks and reported `docs/tutorials` as
  missing, so the published site would have had no tutorials at all.

### Documentation

- A full documentation site, with all eight tutorials rendered with their
  outputs: https://fittingtransientswithdiscs.github.io/FitTeD/
