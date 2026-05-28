"""
Compute and animate the energy profile of a time-evolved state.

Resolves the total energy E = <H> two ways at every time step and renders a GIF
showing how both profiles evolve in time:

  * E(mode): energy per field mode k_m (with the bare-atom energy at k = 0).
  * E(excitation): energy per photon excitation number nu = sum_m n_m.

Both profiles sum to <H> at every frame.

Configurable via command-line key=value pairs (mirrors the other scripts):
  g            coupling strength        (default 0.05)
  atom         initial atom state       (default 'e'; also 'g', '+', '-')
  photon       photon state type        (default 'coherent'; also 'number')
  alpha        coherent amplitude       (default 1.0, used when photon=coherent)
  n            photon number            (default 1, used when photon=number)
  N            excitation cap           (default 2)
  truncation   basis type               (default 'full+totalcap')
  modes        number of cavity modes   (default 16)
  length       cavity length            (default 20)
  t            total evolution time     (default 20.0)
  dt           time step                (default 0.1)
  RWA          rotating-wave approx     (default False)
  out          output directory         (default results/energy_profile_<auto>)
  fps          GIF frames per second    (default 15)
  stride       use every k-th time step (default chosen for ~150 frames)

Examples:
  python energy_profile.py modes=32 N=2 g=0.05 atom=e photon=coherent alpha=1 t=20
  python energy_profile.py modes=16 N=3 photon=number n=1 truncation=full t=30
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from utilities import Config, NumberState, CoherentState
from simulation import Simulation


def run(g, atom, photon_type, alpha, n, N, truncation, modes, L, t, dt, RWA):
    state_type = CoherentState(alpha) if photon_type == "coherent" else NumberState(n)
    config = Config(
        modes=modes,
        length=L,
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
    return sim


def main(**kwargs):
    g = float(kwargs.get("g", 0.05))
    atom = kwargs.get("atom", "e")
    photon_type = kwargs.get("photon", "coherent")
    alpha = float(kwargs.get("alpha", 1.0))
    n = int(kwargs.get("n", 1))
    N = int(kwargs.get("N", 2))
    truncation = kwargs.get("truncation", "full+totalcap")
    modes = int(kwargs.get("modes", 16))
    L = float(kwargs.get("length", 20))
    t = float(kwargs.get("t", 20.0))
    dt = float(kwargs.get("dt", 0.1))
    rwa_raw = kwargs.get("RWA", "False")
    RWA = rwa_raw if isinstance(rwa_raw, bool) else rwa_raw.lower() in ("true", "1", "yes")
    fps = int(kwargs.get("fps", 15))

    photon_label = f"coherent(a={alpha})" if photon_type == "coherent" else f"number(n={n})"
    print(f"Running truncation={truncation}, modes={modes}, N={N}, g={g}, atom={atom}, "
          f"photon={photon_label}, RWA={RWA}")

    sim = run(g, atom, photon_type, alpha, n, N, truncation, modes, L, t, dt, RWA)

    times = sim.times
    kmodes, E_modes, _, E_atom = sim.compute_energy_profile_modes()      # (T,M), (T,)
    exc_axis, E_exc = sim.compute_energy_profile_excitations()           # (T, nu_max+1)
    # Both decompositions sum to <H>; use the excitation sum as the reference total.
    total = E_exc.sum(axis=1)                                            # (T,)

    # Sort field modes by frequency for a clean spectral curve.
    order = np.argsort(kmodes)
    k_sorted = kmodes[order]
    E_modes = E_modes[:, order]

    # Subsample frames to keep the GIF light.
    stride = int(kwargs.get("stride", max(1, len(times) // 150)))
    frames = range(0, len(times), stride)

    # Output directory.
    default_name = f"energy_profile_{truncation.replace('+', '_')}_M{modes}_N{N}_g{g}"
    out_dir = kwargs.get("out", os.path.join(os.path.dirname(__file__), "..", "results", default_name))
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Static summary: time x mode and time x excitation heatmaps.
    # ------------------------------------------------------------------ #
    fig_s, (axm, axe) = plt.subplots(1, 2, figsize=(12, 4.5))
    pcm = axm.pcolormesh(k_sorted, times, E_modes, shading="auto", cmap="viridis")
    axm.axvline(0.0, color="r", lw=0.8, ls="--", alpha=0.6)
    axm.set_xlabel("mode wave-vector $k_m$")
    axm.set_ylabel("t")
    axm.set_title(r"$E(\mathrm{mode})$  (atom at $k=0$)")
    fig_s.colorbar(pcm, ax=axm, label="energy")

    pce = axe.pcolormesh(exc_axis, times, E_exc, shading="auto", cmap="magma")
    axe.set_xlabel(r"excitation number $\nu$")
    axe.set_ylabel("t")
    axe.set_title(r"$E(\mathrm{excitation})$")
    fig_s.colorbar(pce, ax=axe, label="energy")
    fig_s.tight_layout()
    png_path = os.path.join(out_dir, "energy_profile_heatmaps.png")
    fig_s.savefig(png_path, dpi=150)
    plt.close(fig_s)
    print(f"Static summary saved to {png_path}")

    # ------------------------------------------------------------------ #
    # Animated GIF: two panels evolving in time.
    # ------------------------------------------------------------------ #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Fixed y-limits across all frames.
    def pad(lo, hi):
        if hi == lo:
            hi = lo + 1e-12
        m = 0.08 * (hi - lo)
        return lo - m, hi + m

    mode_all = np.concatenate([E_modes.ravel(), E_atom])
    m1 = pad(float(mode_all.min()), float(mode_all.max()))
    m2 = pad(float(E_exc.min()), float(E_exc.max()))

    (line_modes,) = ax1.plot([], [], "o-", color="C0", ms=4, label="field modes")
    (pt_atom,) = ax1.plot([], [], "s", color="r", ms=9, label="atom (k=0)")
    ax1.set_xlim(min(k_sorted.min(), 0.0) * 1.05, k_sorted.max() * 1.05)
    ax1.set_ylim(*m1)
    ax1.axvline(0.0, color="r", lw=0.8, ls="--", alpha=0.4)
    ax1.set_xlabel("mode wave-vector $k_m$")
    ax1.set_ylabel("energy")
    ax1.set_title(r"$E(\mathrm{mode})$")
    ax1.legend(loc="upper right")

    bars = ax2.bar(exc_axis, np.zeros_like(exc_axis, dtype=float), color="C1")
    ax2.set_xlim(exc_axis.min() - 0.5, exc_axis.max() + 0.5)
    ax2.set_ylim(*m2)
    ax2.set_xlabel(r"excitation number $\nu$")
    ax2.set_ylabel("energy")
    ax2.set_title(r"$E(\mathrm{excitation})$")

    txt = fig.suptitle("")

    def init():
        line_modes.set_data([], [])
        pt_atom.set_data([], [])
        for b in bars:
            b.set_height(0.0)
        return [line_modes, pt_atom, *bars]

    def update(i):
        line_modes.set_data(k_sorted, E_modes[i])
        pt_atom.set_data([0.0], [E_atom[i]])
        for b, h in zip(bars, E_exc[i]):
            b.set_height(h)
        txt.set_text(
            rf"t = {times[i]:6.2f}    "
            rf"$\Sigma E(\mathrm{{mode}})$ = {E_modes[i].sum() + E_atom[i]:.4f}    "
            rf"$\Sigma E(\nu)$ = {E_exc[i].sum():.4f}"
        )
        return [line_modes, pt_atom, txt, *bars]

    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
    gif_path = os.path.join(out_dir, "energy_profile.gif")
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIF saved to {gif_path}")

    # ------------------------------------------------------------------ #
    # Raw data dump.
    # ------------------------------------------------------------------ #
    npz_path = os.path.join(out_dir, "energy_profile.npz")
    np.savez(
        npz_path,
        times=times,
        kmodes=k_sorted,
        E_modes=E_modes,
        E_atom=E_atom,
        exc_axis=exc_axis,
        E_excitation=E_exc,
        total_energy=total,
    )
    print(f"Data saved to {npz_path}")


if __name__ == "__main__":
    main(**dict(arg.lstrip("-").split("=", 1) for arg in sys.argv[1:]))
