# rabi-truncation

Finite multimode Rabi dynamics, photon-space truncation comparisons, and Gaussian-packet scattering. One engine lives in `src/`; callable experiments live in `experiment/` and are controlled by five notebooks in `notebooks/`.

The conventions are `hbar = c = 1`, field frequencies `abs(k)`, and

\[
H = H_0 + D\sum_m (u_m a_m + u_m^* a_m^\dagger)\sigma_x,
\qquad u_m = i f_m e^{-ik_m x_{\rm tls}}/\sqrt L,
\]

with `coupling='sqrt'` giving `f_m=sqrt(abs(k_m))`, or `coupling='flat'` giving `f_m=1`. All migrated Uri notebooks use `sqrt`.

## Install and run

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'
```

Open a notebook in `notebooks/` using this Python environment. Its parameter cell defines the physical model; the next cell displays one informative DataFrame with cavity parameters, mode count and physical bounds, time settings and vector-memory estimates. It never stops a simulation based on a memory threshold. Numerical data generation, postprocessing and plotting have separate cells; every plot is drawn directly in the notebook, and every import is in its first cell. Comments, labels and docstrings are in English.

```python
import numpy as np
from IPython.display import display
from experiment.scattering import run_scattering
from src.experiment import resource_estimation

param_atom = {'omega_0': 1., 'D': .2, 'L': 2*np.pi,
              'x_tls': 0., 'coupling': 'sqrt'}
param_photon = {'k_0': 1., 'sigma_k': .4, 'x_0': -2.}
param_time_evol = {'T': 4., 'dt': .05}
cutoffs = {'ir_cutoff': 0., 'uv_cutoff': 2.}
CTRL_M_EXPLICIT = True
M = 5

estimate = resource_estimation(param_atom, param_time_evol, cutoffs,
                               n_max=2, CTRL_M_EXPLICIT=CTRL_M_EXPLICIT, M=M)
display(estimate)
experiment = run_scattering(param_photon, param_atom, param_time_evol, cutoffs,
                            n_max=2, CTRL_M_EXPLICIT=CTRL_M_EXPLICIT, M=M)

# Reuse or export tables without parsing console output.
observable_history = experiment.observables_dataframe()
observable_summary = experiment.observables_dataframe(summary=True)
# In a notebook: display(estimate, observable_summary)
# Save if needed: observable_history.to_csv('observables.csv')
```

`estimate_config(config)` and `resource_estimation(...)` return the single estimate DataFrame; the notebook displays it explicitly. Memory covers complex vectors only, including the copied coefficient history; sparse matrices, Python objects, solver workspace and figures are excluded. A final-only observable summary has a single `final` column. No feasibility flag or execution gate is applied.

## Momentum control and bases

- `CTRL_M_EXPLICIT=True`: choose `L` and an odd positive `M`. The grid has exactly `M` points, is symmetric, and includes zero. An even `M` raises `ValueError`.
- `CTRL_M_EXPLICIT=False`: choose `L` and `ir_cutoff <= abs(k) <= uv_cutoff`. The count follows from the cutoffs. Zero is included when `ir_cutoff=0`; a positive IR cutoff excludes it by the band definition.

Both controls use `delta_k=2*pi/L`. The estimator reports effective cutoffs, actual mode counts, zero-mode presence, so either control exposes its resulting physical representation. Optional mode selection is an additional restriction; it can produce holes or asymmetric subsets.

| Photon basis | Constraint | Ket dimension |
|---|---|---|
| `truncated` | At most one occupied mode, occupation at most N | `2*(1+M*N)` |
| `full+totalcap` | Total photon number at most N | `2*comb(M+N,N)` |
| `full` | Each mode occupation at most N | `2*(N+1)**M` |

Gaussian preparation uses `exp(-(k-k_0)**2/(4*sigma_k**2))*exp(-i*k*x_0)` on **all selected signed modes**, normalized to coefficients `c_m` with `sum(abs(c_m)**2)=1`. The documented `number` preparation assumes `n=1`. A coherent packet has physical mode amplitudes `alpha*c_m` and theoretical mean photon number `abs(alpha)**2` before basis projection; the code projects and normalizes it on the chosen basis, which can change that mean. Existing higher-`n` number preparation code is unchanged pending cleanup. The documentation derives the coherent state, projection and actual vector coefficients.

## Experiments and fidelity

The notebooks cover TLS evolution, total-cap convergence, a `D` sweep, mode selection, and scattering. Each notebook calls the matching Python module. `Experiment.compute_energy()` returns the total Hamiltonian expectation over the stored states for checking energy conservation.

Each experiment file contains its numerical function and calls the engine explicitly:

```python
experiment = Experiment(config)
experiment.propagate_state(progress=progress)
experiment.compute_observables()
```

The notebooks call these functions with dictionaries or an `ExperimentConfig`, prepare DataFrames and draw figures in their own cells. Parameters live in the notebook; there are no argument parsers or command-line entry points. DataFrames can be saved directly with `to_csv`. The optional scattering CSV export remains in `run_scattering`.

Figure exports also belong to notebooks. Their serif/LaTeX fonts, thin lines, light grids and blue/purple/orange palette follow the local `calibration_twophoton_waveguideQED` notebooks; these settings are visible in the first cell.

Only `F_state` and `F_atom` are calculated. The common `full` space is **implicit**: overlaps align physical momenta and retain every shared Fock configuration, without allocating full-space vectors or projecting/renormalizing onto the intersection. A simulated comparator such as `full+totalcap` is a separate choice. Coherent projections can already disagree at time zero; the notebooks display initial fidelities.

Scattering populations in positive, negative and zero momentum one-photon ground-TLS sectors are reported directly. With an incident packet containing both signs, these populations are channel occupations; identifying an asymptotic transmission/reflection coefficient requires additional physical assumptions explained in the documentation. With the imposed annihilation phase `exp(-i*k*x_tls)` and the positive Fourier reconstruction `exp(i*k*x)`, the plotted interaction coordinate is `-x_tls`; the documentation derives this sign explicitly.

The [mathematical and code documentation](documentation/main.pdf) is in English and organized into three main sections. It derives the grid controls, coherent projection weights, Hamiltonian and symmetries, observables, fidelity embedding and resource costs; documents experiment outputs; and compares the patch with the student implementation, including parameter conversions and validation limits. Source listings have comments tied to their current line ranges. Its [LaTeX source](documentation/main.tex) and [line-comment generator](documentation/build_code_notes.py) are included. Rebuild from the repository root:

```sh
python documentation/build_code_notes.py
latexmk -pdf -cd documentation/main.tex
```
