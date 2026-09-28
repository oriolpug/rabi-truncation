"""Gaussian photon scattering, with the user's original dictionary API."""

import csv
from pathlib import Path

from src.xp_config import ExperimentConfig
from src.experiment import Experiment


def run_scattering(param_photon, param_atom, param_time_evol,
                   cutoffs=None,
                   n_max=3,
                   truncation='full+totalcap',
                   RWA=False,
                   store_state=True,
                   store_results=False,
                   progress=False,
                   CTRL_M_EXPLICIT=False,
                   M=None):
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
    config = ExperimentConfig(param_photon=param_photon,
                              param_atom=param_atom,
                              param_time_evol=param_time_evol,
                              cutoffs=cutoffs,
                              n_max=n_max, truncation=truncation,
                              RWA=RWA, store_state=store_state,
                              CTRL_M_EXPLICIT=CTRL_M_EXPLICIT, M=M)

    experiment = Experiment(config)
    experiment.propagate_state(progress=progress)
    experiment.compute_observables()
    if store_results:
        path = Path(__file__).resolve().parents[1] / 'results' / 'scattering.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(experiment.observables)
            writer.writerows(zip(*experiment.observables.values()))
    return experiment
