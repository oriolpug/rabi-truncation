"""
Simulate time evolution of the atom and plot excitation probability P_e(t).

Configurable via command-line key=value pairs:
  g            coupling strength          (default 0.01)
  atom         initial atom state         (default 'g'; also 'e', '+', '-')
  photon       photon state type          (default 'coherent'; also 'number')
  alpha        coherent amplitude         (default 1.0, used when photon=coherent)
  n            photon number              (default 1, used when photon=number)
  N            excitation cap             (default 3)
  truncation   basis type                 (default 'full'; also 'truncated', 'truncated+atom')
  modes        number of cavity modes     (default 1 for single-mode, set >1 for multi-mode)
  t            total evolution time       (default 50.0)
  dt           time step                  (default 0.1)
  RWA          rotating-wave approx       (default False)

Examples:
  python atom_evolution.py g=0.1 atom=e photon=coherent alpha=2 N=5 t=100
  python atom_evolution.py photon=number n=3 truncation=truncated N=4
  python atom_evolution.py truncation=full,truncated,truncated+atom g=0.05 N=3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from utilities import Config, NumberState, CoherentState
from simulation import Simulation


def run_simulation(
    g: float,
    atom: str,
    photon_type: str,
    alpha: float,
    n: int,
    N: int,
    truncation: str,
    modes: int,
    t: float,
    dt: float,
    RWA: bool,
) -> tuple[np.ndarray, list[float]]:
    """Run one simulation and return (times, P_e(t))."""
    state_type = CoherentState(alpha) if photon_type == "coherent" else NumberState(n)

    config = Config(
        modes=modes,
        g=g,
        atom_state=atom,
        state=state_type,
        excitation_cap=N,
        truncation=truncation,
        t=t,
        dt=dt,
        RWA=RWA,
    )

    sim = Simulation(config)
    sim.time_evolve()

    times = sim.times
    pe = sim.compute_excited_probability()
    return times, pe


def main(**kwargs):
    g = float(kwargs.get("g", 0.01))
    atom = kwargs.get("atom", "g")
    photon_type = kwargs.get("photon", "coherent")
    alpha = float(kwargs.get("alpha", 1.0))
    n = int(kwargs.get("n", 1))
    N = int(kwargs.get("N", 3))
    modes = int(kwargs.get("modes", 1))
    t = float(kwargs.get("t", 50.0))
    dt = float(kwargs.get("dt", 0.1))
    rwa_raw = kwargs.get("RWA", "False")
    RWA = rwa_raw if isinstance(rwa_raw, bool) else rwa_raw.lower() in ("true", "1", "yes")

    truncations_raw = kwargs.get("truncation", "full")
    truncations = [tr.strip() for tr in truncations_raw.split(",")]

    photon_label = f"coherent(α={alpha})" if photon_type == "coherent" else f"number(n={n})"

    fig, ax = plt.subplots(figsize=(8, 4))

    for truncation in truncations:
        print(f"Running truncation={truncation}, g={g}, atom={atom}, photon={photon_label}, N={N}, RWA={RWA}")
        times, pe = run_simulation(
            g=g, atom=atom, photon_type=photon_type, alpha=alpha, n=n,
            N=N, truncation=truncation, modes=modes, t=t, dt=dt, RWA=RWA,
        )
        ax.plot(times, np.real(pe), label=truncation)

    ax.set_xlabel("t")
    ax.set_ylabel(r"$P_e(t)$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        rf"Atom excitation probability  |  g={g},  atom$_0$={atom},  photon={photon_label},  N={N},  RWA={RWA}"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(**dict(arg.lstrip("-").split("=", 1) for arg in sys.argv[1:]))
