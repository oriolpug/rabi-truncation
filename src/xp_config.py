"""Shared configuration for scripts and notebooks."""

from dataclasses import dataclass, replace


@dataclass
class ExperimentConfig:
    """Shared physical, basis, grid and solver inputs; no simulation on creation.

    Parameters
    ----------
    param_photon : dict[str, object]
        k_0 (float): signed packet momentum centre; sigma_k (positive float):
        Gaussian width in exp(-(k-k_0)**2/(4*sigma_k**2)); x_0 (float):
        initial spatial centre through exp(-i*k*x_0). state='number'/'coherent';
        n (int) is the number-state photon count and alpha (complex) the
        collective coherent amplitude before projection (mean |alpha|**2).
    param_atom : dict[str, object]
        omega_0 (float): TLS excitation energy above g; D (real float):
        physical scale multiplying V in H=H0+D*V; L (positive float):
        periodic box length; x_tls (float): TLS coordinate in exp(-i*k*x_tls).
        coupling='sqrt'/'flat' sets f_m=sqrt(|k_m|)/1; initial_state is one
        of {'g','e','+','-'}. Natural units hbar=c=1 are used throughout.
    param_time_evol : dict[str, object]
        T and dt (positive floats); optional method, rtol and atol. dt selects
        outputs, while the solver chooses its internal integration steps.
    cutoffs : dict[str, float] or None
        ir_cutoff and uv_cutoff, active only when CTRL_M_EXPLICIT=False.
    n_max : int
        Nonnegative photon cutoff N, interpreted according to truncation.
    truncation : str
        'truncated', 'full+totalcap' (default), or 'full'.
    RWA : bool
        Select excitation-conserving coupling terms when True.
    store_state : bool
        True retains the history; False retains only the final state.
    CTRL_M_EXPLICIT : bool
        True chooses an odd positive M; False derives the count from IR/UV.
    M : int or None
        Explicit base-grid count, ignored under cutoff control.
    mode_selection : bool
        Apply packet/resonance windows after constructing the base grid.
    photon_window, atom_window : float
        Nonnegative window radii in units of sigma_k, default 1.5 and 0.25.

    Notes
    -----
    Dataclass fields are available as attributes; consuming routines validate
    them. replace(config, ...) makes a new dataclass, but dictionaries remain
    shared unless explicitly replaced. with_D copies the atomic dictionary.
    """
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
        """Return a configuration with a new physical coupling without mutating input.

        Parameters
        ----------
        D : float
            Coupling coefficient in H=H0+D*V; validated when H is constructed.

        Returns
        -------
        ExperimentConfig
            New dataclass with a copied param_atom dictionary containing D. Other
            dictionaries remain shared by reference; this method does not run a
            simulation or change the original configuration.
        """
        return replace(self, param_atom={**self.param_atom, "D": D})
