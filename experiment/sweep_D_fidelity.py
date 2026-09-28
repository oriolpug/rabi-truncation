"""Time-mean state/TLS fidelities versus the physical coupling D."""

from dataclasses import replace

import numpy as np

from src.fidelities import fidelities_over_time
from ._common import config_from_args, finish, parser, run


def run_coupling_sweep(config, D_values, schemes=('truncated',), reference='full+totalcap', progress=False):
    """Sweep physical D and compare each candidate with a simulated trajectory.

    Parameters
    ----------
    config : ExperimentConfig
        Shared inputs; D and basis are changed in copies, with histories forced.
    D_values : array_like of float
        Nonempty finite vector (Q,) of physical couplings in H=H0+D*V.
    schemes : iterable[str], optional
        S candidate basis names, default ('truncated',).
    reference : str, optional
        Simulated comparator basis, default 'full+totalcap'. This choice is
        separate from the implicit full basis used to calculate overlaps.
    progress : bool, optional
        Show solver progress for each trajectory.

    Returns
    -------
    dict[str, object]
        D array (Q,), schemes tuple (S,), reference string, and four float
        arrays (S,Q): F_state/F_atom are arithmetic sample means
        sum_j F(t_j)/N_t; _initial arrays contain F(0). These are neither final
        fidelities nor quadrature approximations to an integral time average.
        Identical candidate/comparator schemes reuse the comparator run.
    """
    D_values = np.asarray(D_values, dtype=float)
    if D_values.ndim != 1 or len(D_values) == 0 or not np.isfinite(D_values).all():
        raise ValueError('D_values must be a nonempty finite one-dimensional array')
    schemes = tuple(schemes)
    shape = (len(schemes), len(D_values))
    out = {'D': D_values, 'schemes': schemes, 'reference': reference,
           **{name: np.empty(shape) for name in
              ('F_state', 'F_atom', 'F_state_initial', 'F_atom_initial')}}
    for i, D in enumerate(D_values):
        at_D = replace(config.with_D(float(D)), store_state=True)
        ref = run(replace(at_D, truncation=reference), progress)
        for j, scheme in enumerate(schemes):
            candidate = ref if scheme == reference else run(replace(at_D, truncation=scheme), progress)
            fidelities = fidelities_over_time(candidate, ref)
            for name, values in fidelities.items():
                out[name][j, i] = np.mean(values)
                out[f'{name}_initial'][j, i] = values[0]
    return out


def plot_coupling_sweep(results):
    """Plot sample-mean and initial squared fidelities versus physical D.

    Parameters
    ----------
    results : dict[str, object]
        run_coupling_sweep output with D (Q,) and metrics (S,Q).

    Returns
    -------
    matplotlib.figure.Figure
        Two metric panels; one solid mean curve per scheme and dashed initial
        curves. The figure is returned without display or export.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ('F_state', 'F_atom')):
        for j, scheme in enumerate(results['schemes']):
            ax.plot(results['D'], results[name][j], 'o-', label=scheme)
            ax.plot(results['D'], results[f'{name}_initial'][j], '--', alpha=0.5)
        ax.set(xlabel='D', ylabel=f'time mean {name}', ylim=(0, 1.05))
        ax.legend()
    fig.tight_layout()
    return fig


def main(argv=None):
    """Run the physical-D sweep and export sample-mean/initial metrics.

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
    p.add_argument('--D-values', default='0.02,0.1,0.2')
    p.add_argument('--schemes', default='truncated')
    p.add_argument('--reference', choices=('full', 'full+totalcap', 'truncated'), default='full+totalcap')
    args = p.parse_args(argv)
    config = config_from_args(args)
    from src.experiment import estimate_config
    for scheme in set(args.schemes.split(',')) | {args.reference}:
        estimate_config(replace(config, truncation=scheme))
    results = run_coupling_sweep(config, [float(v) for v in args.D_values.split(',')],
                                 args.schemes.split(','), args.reference, args.progress)
    finish(plot_coupling_sweep(results), args, config, **results)


if __name__ == '__main__':
    main()
