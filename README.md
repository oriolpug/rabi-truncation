# rabi-truncation

Finite multimode Rabi dynamics, photon-space truncation comparisons, and Gaussian-packet scattering. One engine lives in `src/`; callable experiments live in `experiment/` and are controlled by six notebooks in `notebooks/`.

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
python -m pytest tests -q
```

Open a notebook in `notebooks/` using this Python environment. Its parameter cell defines the physical model; its next cell prints the grid and memory estimate before running. Notebook outputs are deliberately empty in the repository.

```python
import numpy as np
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
experiment = run_scattering(param_photon, param_atom, param_time_evol, cutoffs,
                            n_max=2, CTRL_M_EXPLICIT=CTRL_M_EXPLICIT, M=M)
```

## Momentum control and bases

- `CTRL_M_EXPLICIT=True`: choose `L` and an odd positive `M`. The grid has exactly `M` points, is symmetric, and includes zero. An even `M` raises `ValueError`.
- `CTRL_M_EXPLICIT=False`: choose `L` and `ir_cutoff <= abs(k) <= uv_cutoff`. The count follows from the cutoffs. Zero is included when `ir_cutoff=0`; a positive IR cutoff excludes it by the band definition.

Both controls use `delta_k=2*pi/L`. The estimator reports effective cutoffs, actual mode counts, zero-mode presence, and whether the other control can reproduce the same grid. Optional mode selection is an additional restriction; it can produce holes or asymmetric subsets.

| Photon basis | Constraint | Ket dimension |
|---|---|---|
| `truncated` | At most one occupied mode, occupation at most N | `2*(1+M*N)` |
| `full+totalcap` | Total photon number at most N | `2*comb(M+N,N)` |
| `full` | Each mode occupation at most N | `2*(N+1)**M` |

Gaussian preparation uses `exp(-(k-k_0)**2/(4*sigma_k**2))*exp(-i*k*x_0)` on **all selected signed modes**. Number states (`n=0` allowed) and product coherent states are projected and normalized on the chosen basis.

## Experiments and fidelity

The notebooks cover TLS evolution, energy profiles and animation, total-cap convergence, a `D` sweep, mode selection, and scattering. Each notebook calls the matching Python module.

```sh
python -m experiment.scattering --D 0.2 --ctrl-m-explicit --M 5
python -m experiment.sweep_D_fidelity --D-values 0.02,0.1,0.2 --out results/sweep.png
python experiments_Uri/energy_profile.py --D 0.2 --out results/energy.png --gif results/energy.gif
```

The five historical `experiments_Uri/` script paths remain runnable as delegates. Parameters now follow the shared `D`, `L`, `M`, `sigma_k`, `n_max`, `T` conventions and `--option value` CLI. The historical filename `sweep_g_fidelity.py` delegates to `experiment/sweep_D_fidelity.py`.

Only `F_state` and `F_atom` are calculated. The common `full` space is **implicit**: overlaps align physical momenta and retain every shared Fock configuration, without allocating full-space vectors or projecting/renormalizing onto the intersection. A simulated comparator such as `full+totalcap` is a separate choice. Coherent projections can already disagree at time zero; the notebooks display initial fidelities.

Scattering populations in positive, negative and zero momentum one-photon ground-TLS sectors are reported directly. With an incident packet containing both signs, these populations are channel occupations; identifying an asymptotic transmission/reflection coefficient requires additional physical assumptions explained in the documentation. With the imposed annihilation phase `exp(-i*k*x_tls)` and the positive Fourier reconstruction `exp(i*k*x)`, the plotted interaction coordinate is `-x_tls`; the documentation derives this sign explicitly.

The [mathematical and code documentation](documentation/main.pdf) is in English and organized into three main sections. It derives the grid controls, coherent projection weights, Hamiltonian and symmetries, observables, fidelity embedding and resource costs; documents experiment outputs; and compares the patch with the student implementation, including parameter conversions and validation limits. Source listings have comments tied to their current line ranges. Its [LaTeX source](documentation/main.tex) and [line-comment generator](documentation/build_code_notes.py) are included. Rebuild from the repository root:

```sh
python documentation/build_code_notes.py
latexmk -pdf -cd documentation/main.tex
```
