"""Final-state convergence between neighbouring total photon caps."""

from dataclasses import replace

import numpy as np

from src.fidelities import compare_states
from src.experiment import Experiment


def run_cap_convergence(config, caps=(1, 2, 3, 4), progress=False):
    """Compare neighboring total-photon caps in the implicit ambient full basis.

    Parameters
    ----------
    config : ExperimentConfig
        Shared preparation and dynamics; basis/storage are replaced in copies.
    caps : iterable[int], optional
        At least two strictly increasing nonnegative caps N_j; defaults 1..4.
        Number preparation must fit every cap; coherent input is projected
        and normalized independently at each cap.
    progress : bool, optional
        Show solver progress.

    Returns
    -------
    dict[str, object]
        ``caps`` integer array (C,), ``runs`` list of C propagated Experiments;
        F_state/F_atom and their _initial counterparts are float arrays (C-1,).
        Element j compares N_j with N_(j+1), at T or zero respectively. All
        trajectories use 'full+totalcap' and store only the final evolved ket.
        Squared fidelities include initial projection differences; no common
        intersection normalization or ambient full-space allocation is used.
    """
    caps = list(caps)
    if len(caps) < 2 or any(b <= a for a, b in zip(caps, caps[1:])):
        raise ValueError('At least two strictly increasing caps are required')
    runs = []
    for cap in caps:
        current_config = replace(config, n_max=cap, truncation='full+totalcap', store_state=False)
        experiment = Experiment(current_config)
        experiment.propagate_state(progress=progress)
        experiment.compute_observables()
        runs.append(experiment)
    final, initial = [], []
    for a, b in zip(runs, runs[1:]):
        final.append(compare_states(a.result.final_state, a.basis, a.k_tab,
                                    b.result.final_state, b.basis, b.k_tab))
        initial.append(compare_states(a.state0, a.basis, a.k_tab, b.state0, b.basis, b.k_tab))
    return {'caps': np.array(caps), 'runs': runs,
            **{name: np.array([row[name] for row in final]) for name in ('F_state', 'F_atom')},
            **{f'{name}_initial': np.array([row[name] for row in initial])
               for name in ('F_state', 'F_atom')}}
