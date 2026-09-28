"""Shared configuration for scripts and notebooks."""

from dataclasses import dataclass, replace


@dataclass
class ExperimentConfig:
    param_photon: dict
    param_atom: dict
    param_time_evol: dict
    cutoffs: dict | None = None
    n_max: int = 3
    truncation: str = "full+totalcap"
    RWA: bool = False
    store_state: bool = True
    CTRL_M_EXPLICIT: bool = False
    M: int | None = None
    mode_selection: bool = False
    photon_window: float = 1.5
    atom_window: float = 0.25

    def with_D(self, D):
        """Copy the atom dictionary when changing the physical coupling."""
        return replace(self, param_atom={**self.param_atom, "D": D})
