"""Truncation and mode-selection errors relative to one finite trajectory."""

from dataclasses import replace

import numpy as np

from src.fidelities import fidelities_over_time
from ._common import config_from_args, finish, parser, run


def run_mode_selection(config, D_values=(0.02, 0.1, 0.2),
                       schemes=('truncated', 'full+totalcap'),
                       photon_windows=(0.5, 1.5, 3.0), atom_windows=(0.0, 0.5, 1.5),
                       D_heatmap=None, progress=False):
    """Measure truncation and selected-grid differences from an unselected run.

    Parameters
    ----------
    config : ExperimentConfig
        Shared inputs, requiring store_state=True. The simulated comparator
        always uses the unselected 'full+totalcap' basis at the same D.
    D_values : array_like of float, optional
        Couplings (Q,) for the selection-on/off sweep.
    schemes : iterable[str], optional
        S candidate photon bases, default ('truncated', 'full+totalcap').
    photon_windows : array_like of float, optional
        P nonnegative packet-window radii w_p in units of sigma_k.
    atom_windows : array_like of float, optional
        A nonnegative resonance-window radii w_a in units of sigma_k.
    D_heatmap : float or None, optional
        Fixed coupling for the window scan; None uses config.param_atom['D'].
    progress : bool, optional
        Show solver progress for each trajectory.

    Returns
    -------
    dict[str, object]
        D (Q,), schemes (S,), selection flags [False,True], window arrays
        (P,)/(A,), and D_heatmap. F_state/F_atom and their _initial arrays
        have shape (S,2,Q), ordered by scheme, selection flag, then coupling.
        Mean-only _heatmap arrays have shape (S,P,A). Means are arithmetic
        averages over stored output times. Missing selected modes are vacuum
        in the common full embedding, and selected preparations are normalized
        on their own grids, so initial fidelities can already be below one.
    """
    if not config.store_state:
        raise ValueError('Time-mean comparisons require store_state=True')
    D_values = np.asarray(D_values, dtype=float)
    schemes = tuple(schemes)
    pw, aw = np.asarray(photon_windows), np.asarray(atom_windows)
    shape = (len(schemes), 2, len(D_values))
    metrics = {name: np.empty(shape) for name in ('F_state', 'F_atom', 'F_state_initial', 'F_atom_initial')}
    for i, D in enumerate(D_values):
        current = config.with_D(float(D))
        reference = run(replace(current, truncation='full+totalcap', mode_selection=False), progress)
        for j, scheme in enumerate(schemes):
            for selected in (False, True):
                candidate = run(replace(current, truncation=scheme, mode_selection=selected), progress)
                values = fidelities_over_time(candidate, reference)
                for name, series in values.items():
                    metrics[name][j, int(selected), i] = np.mean(series)
                    metrics[f'{name}_initial'][j, int(selected), i] = series[0]
    current = config.with_D(config.param_atom['D'] if D_heatmap is None else D_heatmap)
    reference = run(replace(current, truncation='full+totalcap', mode_selection=False), progress)
    heatmaps = {name: np.empty((len(schemes), len(pw), len(aw))) for name in ('F_state', 'F_atom')}
    for j, scheme in enumerate(schemes):
        for x, photon_window in enumerate(pw):
            for y, atom_window in enumerate(aw):
                candidate = run(replace(current, truncation=scheme, mode_selection=True,
                                        photon_window=float(photon_window), atom_window=float(atom_window)), progress)
                for name, series in fidelities_over_time(candidate, reference).items():
                    heatmaps[name][j, x, y] = np.mean(series)
    return {'D': D_values, 'schemes': schemes, 'selection': np.array([False, True]),
            'photon_windows': pw, 'atom_windows': aw, 'D_heatmap': current.param_atom['D'],
            **metrics, **{f'{name}_heatmap': value for name, value in heatmaps.items()}}


def plot_mode_selection(results):
    """Plot selection-on/off sweeps and fixed-D window scans for both fidelities.

    Parameters
    ----------
    results : dict[str, object]
        run_mode_selection output with sweep arrays (S,2,Q) and maps (S,P,A).

    Returns
    -------
    matplotlib.figure.Figure
        (S+1)-by-2 panels: top row shows sample-mean fidelity against D;
        remaining rows show one basis per window map. Map slices are transposed
        for horizontal photon window and vertical atom window coordinates.
    """
    import matplotlib.pyplot as plt
    schemes = results['schemes']
    fig, axes = plt.subplots(len(schemes) + 1, 2, figsize=(11, 3 * (len(schemes) + 1)), squeeze=False)
    for column, name in enumerate(('F_state', 'F_atom')):
        ax = axes[0, column]
        for j, scheme in enumerate(schemes):
            for selected in (False, True):
                ax.plot(results['D'], results[name][j, int(selected)],
                        'o-' if selected else 'o--', label=f'{scheme}, selection={selected}')
        ax.set(xlabel='D', ylabel=f'time mean {name}')
        ax.legend(fontsize=8)
        for j, scheme in enumerate(schemes):
            ax = axes[j + 1, column]
            image = ax.pcolormesh(results['photon_windows'], results['atom_windows'],
                                  results[f'{name}_heatmap'][j].T, shading='nearest', vmin=0, vmax=1)
            fig.colorbar(image, ax=ax)
            ax.set(xlabel='packet window / sigma_k', ylabel='resonance window / sigma_k',
                   title=f'{scheme}: {name}')
    fig.tight_layout()
    return fig


def main(argv=None):
    """Run selection and window sweeps and export fidelity arrays.

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
    p.add_argument('--photon-windows', default='0.5,1.5,3.0')
    p.add_argument('--atom-windows', default='0.0,0.5,1.5')
    args = p.parse_args(argv)
    config = config_from_args(args)
    from src.experiment import estimate_config
    estimate_config(config)
    estimate_config(replace(config, mode_selection=True))
    results = run_mode_selection(config, [float(v) for v in args.D_values.split(',')],
                                 photon_windows=[float(v) for v in args.photon_windows.split(',')],
                                 atom_windows=[float(v) for v in args.atom_windows.split(',')], progress=args.progress)
    finish(plot_mode_selection(results), args, config, **results)


if __name__ == '__main__':
    main()
