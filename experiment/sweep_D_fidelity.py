"""Time-mean state/TLS fidelities versus the physical coupling D."""

from dataclasses import replace

import numpy as np

from src.fidelities import fidelities_over_time
from src.experiment import Experiment


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
        reference_config = replace(at_D, truncation=reference)
        ref = Experiment(reference_config)
        ref.propagate_state(progress=progress)
        ref.compute_observables()
        for j, scheme in enumerate(schemes):
            if scheme == reference:
                candidate = ref
            else:
                candidate_config = replace(at_D, truncation=scheme)
                candidate = Experiment(candidate_config)
                candidate.propagate_state(progress=progress)
                candidate.compute_observables()
            fidelities = fidelities_over_time(candidate, ref)
            for name, values in fidelities.items():
                out[name][j, i] = np.mean(values)
                out[f'{name}_initial'][j, i] = values[0]
    return out
