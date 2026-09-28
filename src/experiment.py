"""Pure-state dynamics and diagnostics shared by every experiment."""

from math import comb
from numbers import Integral

import numpy as np
import pandas as pd
import qutip

from .fidelities import atom_density_matrix
from .grid import momentum_modes, resolve_grid
from .hamiltonians import Hamiltonian
from .states import FockBasis, initial_state
from .xp_config import ExperimentConfig


def output_times(param_time_evol):
    """Construct requested output times including zero and the exact final time.

    Parameters
    ----------
    param_time_evol : dict[str, float]
        Positive finite ``T`` and ``dt``. dt controls output spacing, not the
        solver's adaptive internal steps.

    Returns
    -------
    numpy.ndarray
        Float array (N_t,), starting at 0 and ending at T. Interior samples
        follow j*dt; the last interval may be shorter. If dt>T, returns [0,T].
        Nonpositive/nonfinite inputs raise ValueError.
    """
    T, dt = float(param_time_evol["T"]), float(param_time_evol["dt"])
    if not np.isfinite([T, dt]).all() or T <= 0 or dt <= 0:
        raise ValueError("T and dt must be finite and positive")
    count = int(np.floor(T / dt + 1e-12))
    if count == 0:
        return np.array([0.0, T])
    times = np.arange(count + 1) * dt
    if times[-1] < T and not np.isclose(times[-1], T, atol=1e-12, rtol=0):
        times = np.append(times, T)
    else:
        times[-1] = T
    return times


def resource_estimation(param_atom, param_time_evol, cutoffs=None,
                        n_max=3, truncation="full+totalcap", store_state=True,
                        CTRL_M_EXPLICIT=False, M=None, param_photon=None,
                        mode_selection=False, photon_window=1.5, atom_window=0.25):
    """Return one informative table of cavity, grid, time and vector-memory inputs.

    Parameters
    ----------
    param_atom : dict[str, object]
        Cavity/TLS inputs: L, omega_0, D, x_tls, coupling and initial_state.
    param_time_evol : dict[str, object]
        Positive T and dt, plus optional solver method and tolerances.
    cutoffs : dict[str, float] or None
        Radial ir_cutoff/uv_cutoff bounds under cutoff control.
    n_max : int
        Nonnegative photon cap N.
    truncation : str
        'truncated', 'full+totalcap' or 'full'.
    store_state : bool
        Estimate the complete history when True, otherwise one final ket.
    CTRL_M_EXPLICIT : bool
        True uses an odd positive M; False derives M from the radial bounds.
    M : int or None
        Explicit base-grid count, ignored under cutoff control.
    param_photon : dict[str, object] or None
        k_0 and sigma_k, needed only for optional mode selection.
    mode_selection : bool
        Apply packet/resonance windows before counting retained modes.
    photon_window, atom_window : float
        Selection radii in units of sigma_k.

    Returns
    -------
    pandas.DataFrame
        One value column indexed by group and parameter. Memory is in GiB:
        one complex128 ket costs 16*d bytes; retained vectors approximate
        twice the stored ket history. Sparse matrices, Python objects,
        solver workspace and figures are excluded. This estimate defines
        no feasibility threshold and never authorizes or blocks a simulation.
    """
    L = float(param_atom["L"])
    spacing = 2 * np.pi / L
    if CTRL_M_EXPLICIT:
        if isinstance(M, bool) or not isinstance(M, Integral) or M <= 0 or M % 2 == 0:
            raise ValueError("M must be a positive odd integer")
        k = spacing * np.arange(-(M // 2), M // 2 + 1)
    else:
        ir, uv = cutoffs["ir_cutoff"], cutoffs["uv_cutoff"]
        lower = max(1, int(np.ceil(ir / spacing - 1e-12)))
        upper = int(np.floor(uv / spacing + 1e-12))
        positive = spacing * np.arange(lower, upper + 1)
        k = np.concatenate((-positive[::-1], [0.] if ir == 0 else [], positive))
    if mode_selection:
        sigma = param_photon["sigma_k"]
        mask = np.zeros(len(k), dtype=bool)
        for centre, radius in ((param_photon["k_0"], photon_window * sigma),
                               (param_atom["omega_0"], atom_window * sigma),
                               (-param_atom["omega_0"], atom_window * sigma)):
            window = abs(k - centre) <= radius + 1e-12
            if not window.any():
                window[np.argmin(abs(k - centre))] = True
            mask |= window
        k = k[mask]
    modes = len(k)
    if truncation == "full+totalcap":
        dimension = 2 * comb(modes + n_max, n_max)
    elif truncation == "truncated":
        dimension = 2 * (1 + modes * n_max)
    elif truncation == "full":
        dimension = 2 * (n_max + 1) ** modes
    else:
        raise ValueError(f"Unknown truncation: {truncation}")
    T, dt = param_time_evol["T"], param_time_evol["dt"]
    count = int(np.floor(T / dt + 1e-12))
    n_times = max(2, count + 1 + int(count * dt < T and
                  not np.isclose(count * dt, T, atol=1e-12, rtol=0)))
    ket_gib = 16 * dimension / 2**30
    history_gib = ket_gib * (n_times if store_state else 1)
    rows = [("Cavity", name, value) for name, value in param_atom.items()]
    rows += [("Modes", "M", modes), ("Modes", "delta_k", spacing),
             ("Modes", "k_min", float(k.min())), ("Modes", "k_max", float(k.max())),
             ("Modes", "IR_effective", float(abs(k).min())),
             ("Modes", "UV_effective", float(abs(k).max()))]
    rows += [("Time", name, value) for name, value in param_time_evol.items()]
    rows += [("Time", "n_outputs", n_times),
             ("Memory", "basis", truncation), ("Memory", "n_max", n_max),
             ("Memory", "dimension", dimension), ("Memory", "store_state", store_state),
             ("Memory", "ket_GiB", ket_gib), ("Memory", "history_GiB", history_gib),
             ("Memory", "retained_vectors_GiB", 2 * history_gib)]
    return pd.DataFrame(rows, columns=["Group", "Parameter", "Value"]).set_index(
        ["Group", "Parameter"])


def estimate_config(config):
    """Return the informative resource table for an ExperimentConfig input.

    Parameters
    ----------
    config : ExperimentConfig
        Physical, grid, photon-cap, output-time and storage inputs.

    Returns
    -------
    pandas.DataFrame
        Cavity parameters, retained mode count and physical bounds, time
        settings and estimated vector memory. Display it in the notebook;
        the decision to run remains with the user.
    """
    return resource_estimation(config.param_atom, config.param_time_evol, config.cutoffs,
                               config.n_max, config.truncation, config.store_state,
                               config.CTRL_M_EXPLICIT, config.M, config.param_photon,
                               config.mode_selection, config.photon_window, config.atom_window)


class Experiment:
    """One finite-basis evolution and its reusable physical diagnostics.

    Attributes
    ----------
    config : ExperimentConfig
        Original configuration; physical dictionaries below are copied.
    base_k_tab, k_tab : numpy.ndarray
        Base and retained signed momenta (M_base,)/(M,). n_modes equals M.
    times : numpy.ndarray
        Requested physical output times (N_t,), including zero and T.
    basis : FockBasis
        B field tuples and d=2*B joint ket components q=2*i+s, s=0:g/1:e.
    hamiltonian : Hamiltonian
        Sparse operators H0 and V; H is the corresponding QuTiP H0+D*V.
    state0 : qutip.Qobj
        Normalized projected initial ket (d,1).
    result : qutip.solver.Result
        Set by propagate_state; stored kets and the final state from sesolve.
    c_g_array, c_e_array : numpy.ndarray or None
        Complex amplitudes C_(i,g/e)(t). Shape (N_t,B) with history or (B,)
        for final-only storage. Initially None; slices share coefficient memory.
    observables : dict[str, numpy.ndarray] or None
        Raw population diagnostics, initially None, then arrays (K,) with
        K=N_t for histories or K=1 for final-only storage.
    """
    def __init__(self, config):
        """Build one grid, basis, Hamiltonian and normalized initial ket.

        Parameters
        ----------
        config : ExperimentConfig
            Complete physical and numerical inputs. Dictionary contents are copied
            to the instance; the original configuration itself is retained.

        Returns
        -------
        None
            Initializes the objects documented on Experiment. Allocates the basis,
            sparse matrices and initial ket immediately, but does not propagate.
            Call estimate_config first when allocation may be large.
        """

        print("Initializing experiment...")

        self.config = config
        self.param_photon, self.param_atom = dict(config.param_photon), dict(config.param_atom)
        self.param_time_evol = dict(config.param_time_evol)
        self.n_max, self.truncation = config.n_max, config.truncation
        self.RWA, self.store_state = config.RWA, config.store_state
        self.base_k_tab, self.k_tab = resolve_grid(config)
        self.n_modes = len(self.k_tab)
        self.times = output_times(self.param_time_evol)
        self.basis = FockBasis(self.n_modes, self.n_max, self.truncation)
        self.photon_numbers = self.basis.photon_numbers
        self.hamiltonian = Hamiltonian(self.basis, self.k_tab, self.param_atom, self.RWA)
        self.H = self.hamiltonian.build_hamiltonian()
        self.state0 = initial_state(self.basis, self.k_tab, self.param_photon, self.param_atom)
        self.c_g_array = self.c_e_array = self.observables = None

        print("Done.")

    def propagate_state(self, progress=False):
        """Solve i*d|psi>/dt=H|psi> and retain raw field/TLS amplitudes.

        Parameters
        ----------
        progress : bool, optional
            Enable the QuTiP tqdm progress bar when True.

        Returns
        -------
        c_g_array, c_e_array : tuple[numpy.ndarray, numpy.ndarray]
            Complex arrays (N_t,B) for histories, or (B,) for final-only storage.
            Entry i is the amplitude of field occupation basis.states[i] with TLS
            g or e. Also stores self.result and these arrays on the instance.

        Notes
        -----
        Default solver is BDF with rtol=1e-9, atol=1e-11, overridable through
        param_time_evol. normalize_output=False retains numerical norm drift.
        The final ket is always stored. c_g/c_e are alternating views of a copied
        joint coefficient array, in addition to QuTiP's retained state objects.
        """

        print("Starting propagation...")

        self.result = qutip.sesolve(self.H, self.state0, self.times, options={
            "method": self.param_time_evol.get("method", "bdf"),
            "store_states": self.store_state, 
            "store_final_state": True,
            "normalize_output": False, 
            "rtol": self.param_time_evol.get("rtol", 1e-9),
            "atol": self.param_time_evol.get("atol", 1e-11),
            "progress_bar": "tqdm" if progress else False})
        vectors = np.array([s.full()[:, 0] for s in self.result.states]) if self.store_state \
            else self.result.final_state.full()[:, 0]
        self.c_g_array, self.c_e_array = vectors[..., 0::2], vectors[..., 1::2]

        print("Done.")

        return self.c_g_array, self.c_e_array

    def _one_photon_indices(self):
        """Locate one-photon occupations and map them to physical field modes.

        Parameters
        ----------
        None
            Uses basis.states and the corresponding total photon_numbers.

        Returns
        -------
        indices, mode_ids : tuple[numpy.ndarray, numpy.ndarray]
            Integer arrays (J,), J=M when n_max>=1 and zero otherwise. indices
            are field-occupation indices i (not joint indices q); mode_ids gives
            the sole occupied mode m of each tuple n=e_m. No basis order is assumed.
        """
        indices = np.flatnonzero(self.photon_numbers == 1)
        mode_ids = np.array([np.argmax(self.basis.states[i]) for i in indices], dtype=int)
        return indices, mode_ids

    def compute_observables(self):
        """Compute raw norm, TLS excitation and field-sector populations after evolution.

        Parameters
        ----------
        None
            Uses propagated c_g/c_e arrays and the retained occupation set.

        Returns
        -------
        dict[str, numpy.ndarray]
            Real arrays (K,), K=N_t or 1 for final-only. Includes time,
            norm=sum_(i,s)|C_(i,s)|**2, p_excited=sum_i|C_(i,e)|**2, and
            p_n_photon=sum_(i:sum(n_i)=n,s)|C_(i,s)|**2 for every retained n.
            p_transmitted/p_reflected/p_zero_1g sum |C_(e_m,g)|**2 over k_m>0,
            k_m<0, or k_m=0. These directional entries use only the one-photon,
            ground-TLS sector. Stores the dictionary as self.observables.

        Notes
        -----
        Requires propagate_state. No norm division is applied: sum_n p_n_photon
        equals norm, not necessarily exactly one. Directional populations are
        raw channel occupations, not automatically asymptotic scattering ratios.
        """

        print("Computing observables ...")

        if self.c_g_array is None:
            raise RuntimeError("Call propagate_state first")
        ground, excited = np.atleast_2d(self.c_g_array), np.atleast_2d(self.c_e_array)
        probability = np.abs(ground) ** 2 + np.abs(excited) ** 2
        indices, modes = self._one_photon_indices()
        self.observables = {"time": self.times if self.store_state else self.times[-1:],
                            "norm": probability.sum(axis=1),
                            "p_excited": (np.abs(excited) ** 2).sum(axis=1)}
        for name, mask in (("p_transmitted", self.k_tab[modes] > 0),
                           ("p_reflected", self.k_tab[modes] < 0)):
            self.observables[name] = (np.abs(ground[:, indices[mask]]) ** 2).sum(axis=1)
            
        for n in range(int(self.photon_numbers.max()) + 1):
            self.observables[f"p_{n}_photon"] = probability[:, self.photon_numbers == n].sum(axis=1)

        print("Done.")
        
        return self.observables

    def observables_dataframe(self, summary=False):
        """Return the stored population diagnostics as a pandas DataFrame.

        Parameters
        ----------
        summary : bool, optional
            False returns the complete history indexed by physical time.
            True returns observable rows and initial/final columns; final-only
            storage returns a single final column, never a false initial one.

        Returns
        -------
        pandas.DataFrame
            Raw norm and populations from compute_observables. The summary
            also contains a time row. No probability is renormalized here.

        Notes
        -----
        Requires propagation. Computes observables if absent and leaves the
        numerical arrays intact; table formatting does not round stored data.
        """
        if self.observables is None:
            self.compute_observables()
        frame = pd.DataFrame(self.observables)
        if not summary:
            return frame.set_index("time")
        if self.store_state:
            frame = frame.iloc[[0, -1]].T
            frame.columns = ["initial", "final"]
        else:
            frame = frame.iloc[[-1]].T
            frame.columns = ["final"]
        return frame.rename_axis("observable")

    def one_photon_wavefunction(self, i, x_tab):
        """Reconstruct the one-photon, ground-TLS component on a spatial grid.

        Parameters
        ----------
        i : int
            Stored output index, not physical time. Standard negative indices are
            accepted for histories; final-only storage uses its final coefficients.
        x_tab : array_like of float
            Spatial coordinates (N_x,) in the periodic box.

        Returns
        -------
        numpy.ndarray
            Complex array (N_x,), phi_1g(x,t)=sum_m C_(e_m,g)(t)*exp(i*k_m*x)/sqrt(L).
            This sector is not normalized separately; its box-integrated density
            equals its raw population. With the imposed exp(-i*k*x_tls) annihilation
            phase, the spatial interaction coordinate is -x_tls.
        """
        if self.c_g_array is None:
            raise RuntimeError("Call propagate_state first")
        coefficients = self.c_g_array[i] if self.store_state else self.c_g_array
        indices, modes = self._one_photon_indices()
        return np.exp(1j * np.outer(x_tab, self.k_tab[modes])) @ coefficients[indices] / \
            np.sqrt(self.param_atom["L"])

    def _vectors_for(self, t=None):
        """Resolve a diagnostic time request to the retained joint ket vectors.

        Parameters
        ----------
        t : float or None, optional
            None selects all retained outputs; -1 selects the final state. Other
            values must lie in [0,T] and select the nearest stored time (no
            interpolation). Final-only runs accept None, -1 or the final time only.

        Returns
        -------
        list[numpy.ndarray]
            Raw complex vectors (d,), in joint q=2*i+s order. Length N_t for None
            with a history, otherwise one. Requires propagation; an unavailable
            earlier state or out-of-range time raises ValueError.
        """
        if self.c_g_array is None:
            raise RuntimeError("Call propagate_state first")
        if not self.store_state:
            if t is not None and t != -1 and not np.isclose(t, self.times[-1]):
                raise ValueError("Only the final state was stored")
            return [self.result.final_state.full()[:, 0]]
        if t is None:
            return [s.full()[:, 0] for s in self.result.states]
        if t != -1 and not 0 <= t <= self.times[-1]:
            raise ValueError("Requested time lies outside the stored interval")
        i = -1 if t == -1 else int(np.argmin(abs(self.times - t)))
        return [self.result.states[i].full()[:, 0]]

    def compute_atom_density_matrix(self, t=None):
        """Return the raw field trace rho_TLS(t)=Tr_field |psi(t)><psi(t)|.

        Parameters
        ----------
        t : float or None, optional
            None requests all retained samples; -1 requests final; an in-range
            time uses the nearest retained output. See _vectors_for.

        Returns
        -------
        list[numpy.ndarray] or numpy.ndarray
            None returns K complex matrices (2,2); an explicit time returns one.
            TLS order is (g,e); rho_(s,s')=sum_i C_(i,s)*C_(i,s').conj().
            Trace equals the raw squared ket norm. Final-only with None gives a
            one-element list rather than a reconstructed history.
        """
        matrices = [atom_density_matrix(v) for v in self._vectors_for(t)]
        return matrices if t is None else matrices[0]

    def compute_excited_probability(self, t=None):
        """Evaluate the raw excited-state population P_e=sum_i |C_(i,e)|**2.

        Parameters
        ----------
        t : float or None, optional
            All retained outputs for None, final for -1, or nearest stored output
            to an in-range physical time. Final-only earlier requests are rejected.

        Returns
        -------
        numpy.ndarray or float
            Real array (K,) for None or scalar for an explicit time. K=N_t with
            history and 1 otherwise. No ket-norm division is applied.
        """
        values = np.array([atom_density_matrix(v)[1, 1].real for v in self._vectors_for(t)])
        return values if t is None else float(values[0])


    def compute_energy(self, t=None):
        """Evaluate the raw expectation of the same total Hamiltonian used for dynamics.

        Parameters
        ----------
        t : float or None, optional
            None for all retained outputs, -1 for final, or nearest retained output
            to an in-range physical time. No earlier final-only state is available.

        Returns
        -------
        numpy.ndarray or float
            E(t)=Re[psi(t)^dagger*(H0+D*V)*psi(t)], as a real array (K,) for None
            or scalar otherwise. This retains the raw ket norm and the atomic
            ground energy convention zero. For static Hermitian H, exact evolution
            has dE/dt=0; numerical drift is a diagnostic of the solver.
        """
        matrix = self.hamiltonian.H0 + self.param_atom["D"] * self.hamiltonian.V
        values = np.array([np.vdot(v, matrix @ v).real for v in self._vectors_for(t)])
        return values if t is None else float(values[0])
