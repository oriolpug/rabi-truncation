"""Truncation and mode-selection errors relative to one finite trajectory."""

from dataclasses import replace

import numpy as np

from src.fidelities import fidelities_over_time
from src.experiment import Experiment


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
        reference_config = replace(current, truncation='full+totalcap', mode_selection=False)
        reference = Experiment(reference_config)
        reference.propagate_state(progress=progress)
        reference.compute_observables()
        for j, scheme in enumerate(schemes):
            for selected in (False, True):
                candidate_config = replace(current, truncation=scheme, mode_selection=selected)
                candidate = Experiment(candidate_config)
                candidate.propagate_state(progress=progress)
                candidate.compute_observables()
                values = fidelities_over_time(candidate, reference)
                for name, series in values.items():
                    metrics[name][j, int(selected), i] = np.mean(series)
                    metrics[f'{name}_initial'][j, int(selected), i] = series[0]
    current = config.with_D(config.param_atom['D'] if D_heatmap is None else D_heatmap)
    reference_config = replace(current, truncation='full+totalcap', mode_selection=False)
    reference = Experiment(reference_config)
    reference.propagate_state(progress=progress)
    reference.compute_observables()
    heatmaps = {name: np.empty((len(schemes), len(pw), len(aw))) for name in ('F_state', 'F_atom')}
    for j, scheme in enumerate(schemes):
        for x, photon_window in enumerate(pw):
            for y, atom_window in enumerate(aw):
                candidate_config = replace(current, truncation=scheme, mode_selection=True,
                                           photon_window=float(photon_window), atom_window=float(atom_window))
                candidate = Experiment(candidate_config)
                candidate.propagate_state(progress=progress)
                candidate.compute_observables()
                for name, series in fidelities_over_time(candidate, reference).items():
                    heatmaps[name][j, x, y] = np.mean(series)
    return {'D': D_values, 'schemes': schemes, 'selection': np.array([False, True]),
            'photon_windows': pw, 'atom_windows': aw, 'D_heatmap': current.param_atom['D'],
            **metrics, **{f'{name}_heatmap': value for name, value in heatmaps.items()}}
