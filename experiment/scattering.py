"""Gaussian photon scattering, with the user's original dictionary API."""

import csv
from pathlib import Path

import numpy as np

from src.xp_config import ExperimentConfig
from ._common import config_from_args, finish, parser, run


def run_scattering(param_photon, param_atom, param_time_evol, cutoffs=None,
                   n_max=3, truncation='full+totalcap', RWA=False, store_state=True,
                   store_results=False, progress=False, CTRL_M_EXPLICIT=False, M=None):
    """Propagate Gaussian-photon scattering through the original dictionary API.

    Parameters
    ----------
    param_photon : dict[str, object]
        k_0, sigma_k, x_0 and optional state/n/alpha preparation inputs.
    param_atom : dict[str, object]
        omega_0, D, L, x_tls, coupling ('sqrt'/'flat') and initial_state.
    param_time_evol : dict[str, object]
        T/dt and optional method/rtol/atol solver settings.
    cutoffs : dict[str, float] or None, optional
        ir_cutoff and uv_cutoff when explicit-M control is inactive.
    n_max : int, optional
        Nonnegative photon cap, default 3.
    truncation : str, optional
        Retained photon basis, default 'full+totalcap'.
    RWA : bool, optional
        Keep only excitation-conserving terms when True.
    store_state : bool, optional
        Retain output history (True) or only the final ket (False).
    store_results : bool, optional
        Write/overwrite repository results/scattering.csv with raw observables.
    progress : bool, optional
        Show solver progress.
    CTRL_M_EXPLICIT : bool, optional
        Choose exact odd M when True; choose IR/UV bounds otherwise.
    M : int or None, optional
        Explicit base-grid count, ignored under cutoff control.

    Returns
    -------
    Experiment
        Propagated object with raw observables, kets and diagnostics. Positive,
        negative and zero-momentum packet amplitudes are all retained.
        observables_dataframe(summary=True) exposes the endpoint table.
    """
    config = ExperimentConfig(param_photon, param_atom, param_time_evol, cutoffs,
                              n_max, truncation, RWA, store_state,
                              CTRL_M_EXPLICIT=CTRL_M_EXPLICIT, M=M)
    experiment = run(config, progress)
    if store_results:
        path = Path(__file__).resolve().parents[1] / 'results' / 'scattering.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(experiment.observables)
            writer.writerows(zip(*experiment.observables.values()))
    return experiment


def plot_scattering(experiment):
    """Plot the one-photon spatial density and raw scattering-sector populations.

    Parameters
    ----------
    experiment : Experiment
        Propagated object with compute_observables already called; history
        gives initial/middle/final snapshots, final-only gives one snapshot.

    Returns
    -------
    matplotlib.figure.Figure
        Two panels: |phi_1g(x,t)|**2 over one periodic box and populations
        versus time. Fourier reconstruction uses exp(+i*k*x)/sqrt(L), so the
        exp(-i*k*x_tls) Hamiltonian convention places the marker at -x_tls.
        Directional populations describe the one-photon, ground-TLS sector;
        they are not divided by incident flux or total ket norm.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    times = experiment.observables['time']
    x = np.linspace(-experiment.param_atom['L'] / 2, experiment.param_atom['L'] / 2, 500, endpoint=False)
    for i in sorted({0, len(times) // 2, len(times) - 1}):
        phi = experiment.one_photon_wavefunction(i, x)
        axes[0].plot(x, abs(phi) ** 2, label=f't={times[i]:.2f}')
    # With u ~ exp(-ik*x_tls) and phi ~ exp(+ik*x), the spatial coupling is at -x_tls.
    axes[0].axvline(-experiment.param_atom['x_tls'], color='black', linestyle=':',
                    label='interaction coordinate (-x_tls)')
    axes[0].set(xlabel='x', ylabel=r'$|\phi_{1,g}(x,t)|^2$')
    for name in ('p_transmitted', 'p_reflected', 'p_zero_1g', 'p_excited', 'norm'):
        axes[1].plot(times, experiment.observables[name], label=name)
    axes[1].set(xlabel='t', ylabel='population')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    return fig


def main(argv=None):
    """Run dictionary-API scattering and export raw population curves.

    Parameters
    ----------
    argv : list[str] or None, optional
        CLI arguments without the program name. None reads sys.argv[1:].

    Returns
    -------
    None
        Parses parameters, displays resource tables, simulates and plots.
        --out saves the figure plus a same-stem NPZ; otherwise displays it.
        Argument parsing may raise SystemExit for --help or invalid input.
    """
    p = parser(__doc__)
    args = p.parse_args(argv)
    config = config_from_args(args)
    from src.experiment import estimate_config
    estimate_config(config)
    experiment = run_scattering(config.param_photon, config.param_atom, config.param_time_evol,
                                config.cutoffs, config.n_max, config.truncation, config.RWA,
                                progress=args.progress, CTRL_M_EXPLICIT=config.CTRL_M_EXPLICIT, M=config.M)
    finish(plot_scattering(experiment), args, config, **experiment.observables)


if __name__ == '__main__':
    main()
