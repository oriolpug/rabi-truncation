"""
Sweep the coupling g and compare truncation schemes against the full basis.

For excitation_cap = 4, runs three simulations per g value (full, truncated,
truncated+atom) with otherwise identical parameters, then computes the
time-averaged state fidelity of each truncation against the full simulation.
"""

import argparse
import sys
import os
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utilities import Config, NumberState  # noqa: E402
from simulation import Simulation  # noqa: E402
from states import StateFull, StateTruncated, StateAtom  # noqa: E402
from fidelities import fidelity_statevector  # noqa: E402


STATE_CLS = {
    "full": StateFull,
    "truncated": StateTruncated,
    "truncated+atom": StateAtom,
}


def run(config: Config) -> Simulation:
    sim = Simulation(config)
    sim.time_evolve()
    return sim


def wrap_states(sim: Simulation):
    """Wrap each qutip result state back into the matching State subclass."""
    cls = STATE_CLS[sim.config.truncation]
    return [cls.from_vector(sim.config, s.full()[:, 0]) for s in sim.result.states]


def time_averaged_fidelity(states_a, states_b) -> float:
    fids = [fidelity_statevector(a, b) for a, b in zip(states_a, states_b)]
    return float(np.mean(fids))


def sweep(base_config: Config, g_values: np.ndarray):
    fid_trunc = np.empty_like(g_values, dtype=float)
    fid_atom = np.empty_like(g_values, dtype=float)

    for i, g in enumerate(g_values):
        print(f"\n[{i+1}/{len(g_values)}] g = {g:.4g}")

        full_states = wrap_states(run(replace(base_config, g=float(g), truncation="full")))
        trunc_states = wrap_states(run(replace(base_config, g=float(g), truncation="truncated")))
        atom_states = wrap_states(run(replace(base_config, g=float(g), truncation="truncated+atom")))

        fid_trunc[i] = time_averaged_fidelity(trunc_states, full_states)
        fid_atom[i] = time_averaged_fidelity(atom_states, full_states)
        print(f"    truncated      vs full: {fid_trunc[i]:.6f}")
        print(f"    truncated+atom vs full: {fid_atom[i]:.6f}")

    return fid_trunc, fid_atom


def plot(g_values, fid_trunc, fid_atom, out_path: Path, modes: int, cap: int):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(g_values, fid_trunc, "o-", label="truncated vs full")
    ax.plot(g_values, fid_atom, "s-", label="truncated+atom vs full")
    ax.set_xscale("log")
    ax.set_xlabel("coupling g")
    ax.set_ylabel("time-averaged fidelity")
    ax.set_title(f"Truncation fidelity vs coupling (modes={modes}, N={cap})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", type=int, default=3,
                   help="Number of photon modes (full dim = 2*(N+1)^M; default 3).")
    p.add_argument("--cap", type=int, default=4, help="Excitation cap N (default 4).")
    p.add_argument("--g-min", type=float, default=1e-3)
    p.add_argument("--g-max", type=float, default=3e-1)
    p.add_argument("--n-points", type=int, default=10)
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--out", type=str,
                   default=f"/results/fidelity_vs_g.png")
    args = p.parse_args()

    base_config = Config(
        modes=args.modes,
        excitation_cap=args.cap,
        t=args.t,
        dt=args.dt,
        state=NumberState(1),
        atom_state="g",
    )

    g_values = np.logspace(np.log10(args.g_min), np.log10(args.g_max), args.n_points)
    fid_trunc, fid_atom = sweep(base_config, g_values)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(g_values, fid_trunc, fid_atom, out, args.modes, args.cap)

    np.savez(out.with_suffix(".npz"),
             g=g_values, fid_truncated=fid_trunc, fid_truncated_atom=fid_atom)
    print(f"Data saved to {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
