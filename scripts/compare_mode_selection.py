"""
Quantify the error introduced by photon-mode selection.

For each truncation scheme ("Hamiltonian"), compares the mode-selected time evolution
against the full (all-modes) one of the *same* physical simulation, using the wave-vector-
aligned state fidelity (``fidelities.fidelity_mode_selection``), time-averaged over the
evolution (same convention as ``sweep_g_fidelity.py``).

Two outputs, both produced per scheme:
  1. g-sweep:   time-avg fidelity (selected vs full) vs coupling g, at the default windows.
  2. 2D map:    time-avg fidelity over the two selection-window widths -- photon-window
                half-width around k0 (x) and atom-window half-width around w_atom (y),
                in units of sigma_photon -- as colour, at a fixed g.

Configurable via key=value pairs:
  modes     base mode count        (default 16; the no-selection reference must be tractable)
  length    cavity length          (default 20)
  N         excitation cap         (default 3; use 4 for a finer Fock space)
  t, dt     evolution time / step  (default 20.0 / 0.1)
  photon    photon state type      (default coherent; also number)
  alpha     coherent amplitude     (default 1.0)
  n         Fock number            (default 1, used when photon=number)
  atom      initial atom state     (default g)
  schemes   comma-separated list   (default truncated,truncated+atom,full+totalcap)
  g         fixed coupling for the heatmap          (default 0.1)
  g_min,g_max,g_points   g-sweep range/points       (default 1e-3, 0.3, 10)
  pw_min,pw_max          photon-window range        (default 0.5, 3.0)
  aw_min,aw_max          atom-window range          (default 0.0, 1.5)
  grid      heatmap resolution per axis             (default 7)
  do        which parts to run: sweep,heatmap,both  (default both)
  out       output directory       (default results/compare_mode_selection_<auto>)

Examples:
  python compare_mode_selection.py N=3 modes=16
  python compare_mode_selection.py N=4 modes=16 schemes=full+totalcap grid=9 do=heatmap
"""

import sys
import os
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utilities import Config, NumberState, CoherentState
from simulation import Simulation
from states import StateFull, StateTruncated, StateAtom, StateTotalCap
from fidelities import fidelity_mode_selection


STATE_CLS = {
    "full": StateFull,
    "full+totalcap": StateTotalCap,
    "truncated": StateTruncated,
    "truncated+atom": StateAtom,
}

DEFAULT_PHOTON_WINDOW = 1.5
DEFAULT_ATOM_WINDOW = 0.25


def build_config(scheme, g, modes, length, N, t, dt, state_type, atom,
                 mode_selection=False, photon_window=DEFAULT_PHOTON_WINDOW,
                 atom_window=DEFAULT_ATOM_WINDOW):
    return Config(
        modes=modes,
        length=length,
        g=g,
        atom_state=atom,
        state=state_type,
        excitation_cap=N,
        truncation=scheme,
        t=t,
        dt=dt,
        mode_selection=mode_selection,
        photon_window=photon_window,
        atom_window=atom_window,
    )


def run(config):
    sim = Simulation(config)
    sim.time_evolve()
    return sim


def wrap_states(sim):
    """Wrap each qutip result state back into the matching State subclass."""
    cls = STATE_CLS[sim.config.truncation]
    return [cls.from_vector(sim.config, s.full()[:, 0]) for s in sim.result.states]


def timeavg_fidelity(sel_sim, ref_sim):
    """Time-averaged wave-vector-aligned fidelity between a selected and a full sim."""
    sel, ref = wrap_states(sel_sim), wrap_states(ref_sim)
    return float(np.mean([fidelity_mode_selection(a, b) for a, b in zip(sel, ref)]))


# --------------------------------------------------------------------------- #
# (1) g-sweep
# --------------------------------------------------------------------------- #

def g_sweep(schemes, g_values, base):
    """Return dict[scheme] -> fidelity array over g_values."""
    out = {s: np.empty(len(g_values), dtype=float) for s in schemes}
    for scheme in schemes:
        for i, g in enumerate(g_values):
            ref = run(build_config(scheme, float(g), mode_selection=False, **base))
            sel = run(build_config(scheme, float(g), mode_selection=True, **base))
            out[scheme][i] = timeavg_fidelity(sel, ref)
            print(f"  [sweep] {scheme:<16} g={g:.4g}  "
                  f"modes {ref.config.modes}->{sel.config.modes}  F={out[scheme][i]:.6f}")
    return out


def plot_g_sweep(g_values, results, out_path, title_suffix):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {"truncated": "o-", "truncated+atom": "s-", "full+totalcap": "^-", "full": "d-"}
    for scheme, fid in results.items():
        ax.plot(g_values, fid, markers.get(scheme, "o-"), label=scheme)
    ax.set_xscale("log")
    ax.set_xlabel("coupling g")
    ax.set_ylabel("time-avg fidelity (selected vs full)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Mode-selection fidelity vs g\n{title_suffix}", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


# --------------------------------------------------------------------------- #
# (2) 2D window-width heatmap (per scheme)
# --------------------------------------------------------------------------- #

def window_heatmap(scheme, pw_axis, aw_axis, g, base):
    """Return (fidelity[ny, nx], kept_modes[ny, nx]) over (photon_window, atom_window)."""
    ref = run(build_config(scheme, g, mode_selection=False, **base))
    n_full = ref.config.modes
    F = np.empty((len(aw_axis), len(pw_axis)), dtype=float)
    kept = np.empty((len(aw_axis), len(pw_axis)), dtype=int)
    for ix, pw in enumerate(pw_axis):
        for iy, aw in enumerate(aw_axis):
            sel = run(build_config(scheme, g, mode_selection=True,
                                   photon_window=float(pw), atom_window=float(aw), **base))
            F[iy, ix] = timeavg_fidelity(sel, ref)
            kept[iy, ix] = sel.config.modes
            print(f"  [map:{scheme}] photon_window={pw:.3g} atom_window={aw:.3g}  "
                  f"modes {n_full}->{kept[iy, ix]}  F={F[iy, ix]:.6f}")
    return F, kept


def plot_heatmap(scheme, pw_axis, aw_axis, F, g, out_path, title_suffix):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    pcm = ax.pcolormesh(pw_axis, aw_axis, F, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel(r"photon-window half-width around $k_0$  [$\sigma$]")
    ax.set_ylabel(r"atom-window half-width around $w_{atom}$  [$\sigma$]")
    ax.set_title(f"{scheme}: fidelity vs selection windows  (g={g})\n{title_suffix}", fontsize=10)
    fig.colorbar(pcm, ax=ax, label="time-avg fidelity (selected vs full)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


# --------------------------------------------------------------------------- #

def main(**kwargs):
    modes = int(kwargs.get("modes", 16))
    length = float(kwargs.get("length", 20))
    N = int(kwargs.get("N", 3))
    t = float(kwargs.get("t", 20.0))
    dt = float(kwargs.get("dt", 0.1))
    photon_type = kwargs.get("photon", "coherent")
    alpha = float(kwargs.get("alpha", 1.0))
    n = int(kwargs.get("n", 1))
    atom = kwargs.get("atom", "g")
    state_type = CoherentState(alpha) if photon_type == "coherent" else NumberState(n)

    schemes = [s.strip() for s in kwargs.get(
        "schemes", "truncated,truncated+atom,full+totalcap").split(",")]

    g_heat = float(kwargs.get("g", 0.1))
    g_min = float(kwargs.get("g_min", 1e-3))
    g_max = float(kwargs.get("g_max", 0.3))
    g_points = int(kwargs.get("g_points", 10))

    pw_min = float(kwargs.get("pw_min", 0.5))
    pw_max = float(kwargs.get("pw_max", 3.0))
    aw_min = float(kwargs.get("aw_min", 0.0))
    aw_max = float(kwargs.get("aw_max", 1.5))
    grid = int(kwargs.get("grid", 7))

    do = kwargs.get("do", "both")
    do_sweep = do in ("both", "sweep")
    do_heatmap = do in ("both", "heatmap")

    photon_label = f"coherent(a={alpha})" if photon_type == "coherent" else f"number(n={n})"
    title_suffix = f"(modes={modes}, N={N}, {photon_label}, atom={atom})"

    base = dict(modes=modes, length=length, N=N, t=t, dt=dt, state_type=state_type, atom=atom)

    default_name = f"compare_mode_selection_M{modes}_N{N}"
    out_dir = kwargs.get("out", os.path.join(os.path.dirname(__file__), "..", "results", default_name))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Schemes: {schemes}  |  {title_suffix}")
    if do_heatmap:
        print(f"Heatmap cost ~ {len(schemes)} x ({grid}x{grid} + 1) = "
              f"{len(schemes) * (grid * grid + 1)} simulations; full+totalcap grows with N/modes.")

    # (1) g-sweep ---------------------------------------------------------------
    if do_sweep:
        print("\n=== g-sweep ===")
        g_values = np.logspace(np.log10(g_min), np.log10(g_max), g_points)
        sweep_results = g_sweep(schemes, g_values, base)
        plot_g_sweep(g_values, sweep_results, os.path.join(out_dir, "fidelity_vs_g.png"), title_suffix)
        np.savez(os.path.join(out_dir, "fidelity_vs_g.npz"), g=g_values,
                 **{scheme.replace('+', '_'): sweep_results[scheme] for scheme in schemes})
        print(f"Data saved to {os.path.join(out_dir, 'fidelity_vs_g.npz')}")

    # (2) 2D window heatmap, one per scheme ------------------------------------
    if do_heatmap:
        print("\n=== window heatmaps ===")
        pw_axis = np.linspace(pw_min, pw_max, grid)
        aw_axis = np.linspace(aw_min, aw_max, grid)
        heat_data = {}
        for scheme in schemes:
            F, kept = window_heatmap(scheme, pw_axis, aw_axis, g_heat, base)
            fname = os.path.join(out_dir, f"window_heatmap_{scheme.replace('+', '_')}.png")
            plot_heatmap(scheme, pw_axis, aw_axis, F, g_heat, fname, title_suffix)
            heat_data[f"F_{scheme.replace('+', '_')}"] = F
            heat_data[f"kept_{scheme.replace('+', '_')}"] = kept
        np.savez(os.path.join(out_dir, "window_heatmaps.npz"),
                 photon_window=pw_axis, atom_window=aw_axis, g=g_heat, **heat_data)
        print(f"Data saved to {os.path.join(out_dir, 'window_heatmaps.npz')}")


if __name__ == "__main__":
    main(**dict(arg.lstrip("-").split("=", 1) for arg in sys.argv[1:]))
