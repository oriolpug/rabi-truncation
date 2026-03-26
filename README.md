# rabi-truncation

Quantum Rabi model simulator for studying how different Hilbert-space truncation schemes affect time evolution fidelity in a two-level atom coupled to a multi-mode photon field.

## Research question

How much does truncating the photon Fock space affect the fidelity of the time-evolved state relative to the full (unconstrained) basis? Three truncation schemes are compared.

## Truncation schemes

| Name | Description | Hilbert-space dim |
|---|---|---|
| `"full"` | All Fock states up to cap per mode | 2(N+1)^M |
| `"truncated"` | Vacuum + single-mode excitations only | 2(MN+1) |
| `"truncated+atom"` | Truncated + atom oscillator mode | 2(MN+1)(N+1) |

where M = number of photon modes, N = excitation cap.

## Repository layout

```
src/
  utilities.py     # Config dataclass, StateType hierarchy, wave-vector grid, purity/entropy
  states.py        # Basis mixins + State classes + state() factory
  hamiltonians.py  # Basis mixins + Hamiltonian classes + hamiltonian() factory
  fidelities.py    # common_basis(), fidelity_statevector()
  simulation.py    # Simulation class — orchestrates H build, state init, sesolve, observables
tests/
  conftest.py
  test_states.py
  test_hamiltonians.py
  test_utilities.py
  test_fidelities.py
```

## Installation

Python 3.12+ required. Install dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy qutip matplotlib
```

## Running tests

```bash
.venv/bin/python -m pytest tests/ -v           # all tests
.venv/bin/python -m pytest tests/test_states.py -v    # single file
```

## Basic usage

```python
from src.utilities import Config, NumberState
from src.simulation import Simulation
from src.fidelities import fidelity_statevector

# Full basis simulation
config_full = Config(modes=4, truncation="full", excitation_cap=3, t=10.0, dt=0.1)
sim_full = Simulation(config_full)
sim_full.time_evolve()

# Truncated basis simulation (same physical parameters)
config_trunc = Config(modes=4, truncation="truncated", excitation_cap=3, t=10.0, dt=0.1)
sim_trunc = Simulation(config_trunc)
sim_trunc.time_evolve()

# Compare fidelity at final time
state_full = sim_full.result.states[-1]
state_trunc = sim_trunc.result.states[-1]
F = fidelity_statevector(state_full, state_trunc)
print(f"Fidelity: {F:.6f}")
```

## Key configuration parameters

| Parameter | Default | Description |
|---|---|---|
| `modes` | 64 | Number of photon modes |
| `length` | 10π | System length (sets wave-vector grid) |
| `g` | 0.01 | Atom-photon coupling strength |
| `w_atom` | 1.0 | Atom transition frequency |
| `excitation_cap` | 3 | Max photons per mode (N) |
| `truncation` | `"full"` | Truncation scheme |
| `RWA` | `False` | Rotating-wave approximation |
| `t` | 10.0 | Total evolution time |
| `dt` | 0.1 | Time step |
| `atom_state` | `"g"` | Initial atom state (`"g"`, `"e"`, `"+"`, `"-"`) |
| `state` | `NumberState` | Initial photon state type (`NumberState`, `CoherentState`) |

## Observables

After calling `time_evolve()`, the `Simulation` object provides:

```python
sim.compute_atom_density_matrix()    # 2×2 reduced density matrix (full time series or at time t)
sim.compute_excited_probability()    # P(excited) over time
sim.compute_entropy()                # von Neumann entropy of atom subsystem
sim.compute_energy()                 # ⟨H⟩ over time
```

## Dependencies

- [NumPy](https://numpy.org/) >= 2.4
- [SciPy](https://scipy.org/) >= 1.17
- [QuTiP](https://qutip.org/) >= 5.2
- [Matplotlib](https://matplotlib.org/) >= 3.10
