"""TLS population and entropy for several photon bases."""

from dataclasses import replace

from ._common import config_from_args, finish, parser, run


def run_atom_evolution(config, schemes=('full+totalcap', 'truncated'), progress=False):
    return {scheme: run(replace(config, truncation=scheme, store_state=True), progress)
            for scheme in schemes}


def plot_atom_evolution(runs):
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
