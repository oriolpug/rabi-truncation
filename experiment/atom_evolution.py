"""TLS population and entropy for several photon bases."""

from dataclasses import replace

from ._common import config_from_args, finish, parser, run


def run_atom_evolution(config, schemes=('full+totalcap', 'truncated'), progress=False):
    """Propagate one TLS preparation under several photon-space constraints.

    Parameters
    ----------
    config : ExperimentConfig
        Shared physical inputs and photon cap; input is not modified.
    schemes : iterable[str], optional
        Basis names; default ('full+totalcap', 'truncated').
    progress : bool, optional
        Show solver progress for each trajectory.

    Returns
    -------
    dict[str, Experiment]
        Scheme -> propagated object, forcing store_state=True in each copy.
        The retained histories support raw P_e(t)=sum_i|C_(i,e)|**2 and
        S_TLS(t)=-Tr(rho_hat*log(rho_hat)), with rho_hat normalized to trace one.
    """
    return {scheme: run(replace(config, truncation=scheme, store_state=True), progress)
            for scheme in schemes}


def plot_atom_evolution(runs):
    """Plot raw TLS excitation and normalized entanglement entropy for each basis.

    Parameters
    ----------
    runs : dict[str, Experiment]
        Propagated full-history objects returned by run_atom_evolution.

    Returns
    -------
    matplotlib.figure.Figure
        Two panels of P_e(t) and S_TLS(t), using natural-log entropy. Builds
        the figure without showing, exporting or rerunning the trajectories.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for scheme, experiment in runs.items():
        axes[0].plot(experiment.times, experiment.compute_excited_probability(), label=scheme)
        axes[1].plot(experiment.times, experiment.compute_entropy(), label=scheme)
    axes[0].set(xlabel='t', ylabel=r'$P_e$')
    axes[1].set(xlabel='t', ylabel=r'$S_{\mathrm{TLS}}$')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    return fig


def main(argv=None):
    """Run per-basis TLS evolution and export excitation curves.

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
    p.add_argument('--schemes', default='full+totalcap,truncated')
    args = p.parse_args(argv)
    config = config_from_args(args)
    from src.experiment import estimate_config
    for scheme in args.schemes.split(','):
        estimate_config(replace(config, truncation=scheme))
    runs = run_atom_evolution(config, args.schemes.split(','), args.progress)
    arrays = {f'{scheme}_P_e': exp.compute_excited_probability() for scheme, exp in runs.items()}
    finish(plot_atom_evolution(runs), args, config, time=next(iter(runs.values())).times, **arrays)


if __name__ == '__main__':
    main()
