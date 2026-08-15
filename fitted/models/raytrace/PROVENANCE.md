# raytrace -- provenance

This subpackage is a **minimal extracted subset** of `kerrgeo`, the closed-form
Kerr photon ray tracer by A. Mummery.  It is vendored here so that FitTeD's
ray-traced disc models have no external dependency beyond numpy and scipy.

## Upstream

* Source: `kerrgeo` (private repository)
* Commit: **<FILL IN: output of `git -C kerrgeo rev-parse --short HEAD`>**
* Date vendored: **<FILL IN>**

Record the commit. Without it you cannot tell, later, whether a defect here
is yours or upstream's.

## What was kept

20 modules, ~4250 lines, out of 53 modules / 15540 lines of non-test upstream
code (27%).  Everything retained is on the execution path of the disc SED
calculation, i.e. the vectorised `make_batch` route through `intensity_trace`.

## What was removed, and why

| Removed | Lines | Reason |
|---|---|---|
| `geodesic.py` | 692 | scalar (non-batch) geodesic classes, used only by `AsymptoticObserver.make_geodesics()` |
| `tphi_radial`, `tphi_quad`, `tphi_elliptic`, `_batch_tphi`, `camera_tphi`, `elliptic_unfolded` | 3107 | photon arrival time / azimuth integration.  `intensity_trace` returns only `r_hit`, `theta_hit`, `g`, `valid` -- no (t, phi) -- so none of this is reachable. |
| `tetrad`, `trace`, `polarized_trace`, `walker_penrose`, `reflected_polarized_trace`, `returning_*` | ~3400 | polarization and returning-radiation pipelines |
| `emission/` (7 modules) | ~1300 | Rayleigh / Chandrasekhar atmosphere models |
| `observers/disk_emitter`, `observers/stationary`, `sources/thermal_disk`, other `targets/*` | ~1500 | unused observer, source and target geometries |

The whole (t,phi) stack was reachable through a single module-level import of
`geodesic` in `observers/asymptotic.py`.  That import is now lazy, inside
`make_geodesics()`, which is the only consumer.

## Verification

`intensity_trace` output was compared against the full upstream package over a
grid of 9 spins (-0.9 to 0.998, including retrograde), 5 inclinations
(5-85 deg) and both source types -- **90 cases, all four returned arrays
(`valid`, `r_hit`, `g`, `theta_hit`) bit-identical**.  `planck_weight` is
bit-identical over 200 energies from 1e-4 to 1e2 keV.

Re-run that comparison before accepting any future re-sync from upstream.

## Known limitations of this subset

* **`a = 0` exactly is not supported.**  The vectorised `make_batch` path does
  not dispatch to the Schwarzschild backend; upstream directs `a = 0` cases to
  `make_geodesics()`, which is not vendored.  FitTeD therefore floors `|a|` --
  see `numerical_disc_model`.  Measured cost of the floor: `a = 1e-8` versus
  `a = 1e-6` differs by <= 3e-6 in `g` and `r_hit` and gives an identical
  `valid` mask, so the substitution is physically indistinguishable.
* Only the vectorised batch path is available.  `make_geodesics()` will raise
  `ImportError` if called.

## Divergences from upstream

1. Absolute `kerrgeo.*` imports rewritten as package-relative, so the
   subpackage works at any nesting depth.
2. `observers/asymptotic.py`: the `geodesic` import made lazy (see above).
3. `sources/kerr_disk.py`: `np.errstate` around the circular-orbit
   four-velocity, suppressing an expected NaN warning for pixels inside the
   marginally-stable orbit.  No numerical change.
4. `__init__.py` reduced to the six symbols FitTeD uses.
