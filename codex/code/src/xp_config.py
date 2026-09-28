from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Parameters of one waveguide scattering calculation."""

    param_photon: dict
    param_atom: dict
    param_time_evol: dict
    cutoffs: dict
    n_max: int = 3
    truncation: str = "full+totalcap"
    RWA: bool = False
    store_state: bool = True
