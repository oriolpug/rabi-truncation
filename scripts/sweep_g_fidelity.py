"""
Sweep the coupling g and compare truncation schemes against the full basis.

The reference ("full") simulation uses the ``full+totalcap`` scheme: all Fock
states with total photon number across modes ≤ N (dim = 2·C(N+M, M)), so the
excitation cap is global, not per-mode.

For each g value, runs three simulations (full+totalcap, truncated,
truncated+atom) with otherwise identical parameters, then computes three
time-averaged fidelities of each truncation against the reference:

  1. F_state    — full state fidelity |<ψ_ref|ψ_trunc>|²
  2. F_atom     — fidelity of the 2x2 atom reduced density matrix
  3. F_photon_b — F_state / F_atom (proxy for the photon-sector fidelity;
                  computing the photon reduced density matrix directly is
                  prohibitively expensive)
"""

import argparse
import sys
import os
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utilities import Config, NumberState, CoherentState  # noqa: E402
from simulation import Simulation  # noqa: E402
from states import StateFull, StateTruncated, StateAtom, StateTotalCap  # noqa: E402
from fidelities import fidelity_statevector  # noqa: E402


STATE_CLS = {
    "full": StateFull,
    "full+totalcap": StateTotalCap,
    "truncated": StateTruncated,
    "truncated+atom": StateAtom,
}

REFERENCE = "full+totalcap"
SCHEMES = ["truncated", "truncated+atom"]


def run(config: Config) -> Simulation:
    sim = Simulation(config)
    sim.time_evolve()
    return sim


def wrap_states(sim: Simulation):
    """Wrap each qutip result state back into the matching State subclass."""
    cls = STATE_CLS[sim.config.truncation]
    return [cls.from_vector(sim.config, s.full()[:, 0]) for s in sim.result.states]


def fidelity_atom(s1, s2) -> float:
    """Fidelity of the 2x2 atom reduced density matrices in the |<·|·>|² convention.

    qutip.fidelity returns Tr √(√ρ σ √ρ); we square it so that for pure states
    it agrees with |<ψ|φ>|² (matching `fidelity_statevector`).
    """
    rho1 = qutip.Qobj(s1.atom_density_matrix())
    rho2 = qutip.Qobj(s2.atom_density_matrix())
    return float(qutip.fidelity(rho1, rho2)) ** 2


def fidelities_over_time(states_a, states_b):
    """Return (F_state, F_atom, F_photon_bound) time series."""
    f_state = np.array([fidelity_statevector(a, b) for a, b in zip(states_a, states_b)])
    f_atom = np.array([fidelity_atom(a, b) for a, b in zip(states_a, states_b)])
    # Avoid div-by-zero: when atom fidelity collapses, the bound is undefined.
    with np.errstate(divide="ignore", invalid="ignore"):
        f_photon = np.where(f_atom > 0, f_state / f_atom, np.nan)
    return f_state, f_atom, f_photon


def sweep(base_config: Config, g_values: np.ndarray):
    """Returns dict[scheme] -> dict[metric] -> array over g."""
    metrics = ["state", "atom", "photon_bound"]
    out = {scheme: {m: np.empty_like(g_values, dtype=float) for m in metrics}
           for scheme in SCHEMES}

    for i, g in enumerate(g_values):
        print(f"\n[{i+1}/{len(g_values)}] g = {g:.4g}")
        ref_states = wrap_states(run(replace(base_config, g=float(g), truncation=REFERENCE)))

        for scheme in SCHEMES:
            sch_states = wrap_states(run(replace(base_config, g=float(g), truncation=scheme)))
            f_state, f_atom, f_photon = fidelities_over_time(sch_states, ref_states)
            out[scheme]["state"][i] = float(np.mean(f_state))
            out[scheme]["atom"][i] = float(np.mean(f_atom))
            out[scheme]["photon_bound"][i] = float(np.nanmean(f_photon))
            print(f"    {scheme:<16} state={out[scheme]['state'][i]:.6f}  "
                  f"atom={out[scheme]['atom'][i]:.6f}  "
                  f"photon~={out[scheme]['photon_bound'][i]:.6f}")

    return out


def _plot_one(g_values, results, metric, ylabel, title, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {"truncated": "o-", "truncated+atom": "s-"}
    for scheme in SCHEMES:
        ax.plot(g_values, results[scheme][metric], markers[scheme],
                label=f"{scheme} vs {REFERENCE}")
    ax.set_xscale("log")
    ax.set_xlabel("coupling g")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def plot_all(g_values, results, base_out: Path, modes: int, cap: int):
    suffix_title = f"(modes={modes}, N={cap})"
    stem, ext = base_out.stem, base_out.suffix or ".png"
    parent = base_out.parent
    plots = [
        ("state",        "time-avg state fidelity",        f"State fidelity vs g {suffix_title}",
         parent / f"{stem}_state{ext}"),
        ("atom",         "time-avg atom fidelity",         f"Atom reduced-DM fidelity vs g {suffix_title}",
         parent / f"{stem}_atom{ext}"),
        ("photon_bound", "F_state / F_atom (photon proxy)", f"Photon-sector fidelity bound vs g {suffix_title}",
         parent / f"{stem}_photon_bound{ext}"),
    ]
    for metric, ylabel, title, path in plots:
        _plot_one(g_values, results, metric, ylabel, title, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", type=int, default=3,
                   help="Number of photon modes (reference dim = 2*C(N+M, M); default 3).")
    p.add_argument("--cap", type=int, default=4,
                   help="Total excitation cap N across all modes (default 4).")
    p.add_argument("--g-min", type=float, default=1e-3)
    p.add_argument("--g-max", type=float, default=3e-1)
    p.add_argument("--n-points", type=int, default=10)
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--out", type=str,
                   default="results/fidelity_vs_g.png")
    p.add_argument("--init-state", choices=["number", "coherent"], default="number",
                   help="Initial photon state type (default: number).")
    p.add_argument("--n", type=int, default=1,
                   help="Fock number for --init-state=number (default: 1).")
    p.add_argument("--alpha", type=str, default="1.0",
                   help="Complex amplitude for --init-state=coherent, "
                        "parsed as a Python complex (e.g. '1+0.5j'). Default: 1.0.")
    args = p.parse_args()

    if args.init_state == "number":
        init_state = NumberState(args.n)
    else:
        init_state = CoherentState(complex(args.alpha))

    base_config = Config(
        modes=args.modes,
        excitation_cap=args.cap,
        t=args.t,
        dt=args.dt,
        state=init_state,
        atom_state="g",
    )

    g_values = np.logspace(np.log10(args.g_min), np.log10(args.g_max), args.n_points)
    results = sweep(base_config, g_values)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_all(g_values, results, out, args.modes, args.cap)

    npz_path = out.with_suffix(".npz")
    np.savez(npz_path, g=g_values,
             **{f"{scheme.replace('+', '_')}_{metric}": results[scheme][metric]
                for scheme in SCHEMES for metric in ("state", "atom", "photon_bound")})
    print(f"Data saved to {npz_path}")


if __name__ == "__main__":
    main()
