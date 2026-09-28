"""TLS population and entropy for several photon bases."""

from dataclasses import replace

from src.experiment import Experiment


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
    runs = {}
    for scheme in schemes:
        current_config = replace(config, truncation=scheme, store_state=True)
        experiment = Experiment(current_config)
        experiment.propagate_state(progress=progress)
        experiment.compute_observables()
        runs[scheme] = experiment
    return runs
