"""Pure-state dynamics and diagnostics shared by every experiment."""

from math import comb

import numpy as np
import qutip

from .fidelities import atom_density_matrix
from .grid import grid_summary, integer, momentum_modes, resolve_grid
from .hamiltonians import Hamiltonian
from .reporting import display_resource_tables, observables_table, resource_tables
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
                        mode_selection=False, photon_window=1.5, atom_window=0.25,
                        print_report=True):
    """Estimate basis dimensions and retained-vector memory before allocation.

    Parameters
    ----------
    param_atom : dict[str, object]
        Contains L (positive float); omega_0 is also used for selected windows.
    param_time_evol : dict[str, float]
        T and dt determine the N_t requested outputs.
    cutoffs : dict[str, float] or None, optional
        ir_cutoff/uv_cutoff radial bounds when explicit-M control is disabled.
    n_max : int, optional
        Nonnegative photon cap N, default 3.
    truncation : {'full+totalcap', 'truncated', 'full'}, optional
        Joint dimensions d=2*binomial(M+N,N), 2*(1+M*N), or 2*(N+1)**M.
    store_state : bool, optional
        True estimates N_t retained kets; False estimates one final ket.
    CTRL_M_EXPLICIT : bool, optional
        True chooses an odd positive M; False uses the cutoff band.
    M : int or None, optional
        Explicit base-grid mode count; ignored under cutoff control.
    param_photon : dict[str, object] or None, optional
        k_0 and sigma_k are required only when mode_selection is enabled.
    mode_selection : bool, optional
        Apply packet/resonance windows before computing the dimension.
    photon_window, atom_window : float, optional
        Window radii as multiples of sigma_k, default 1.5 and 0.25.
    print_report : bool, optional
        Display resource DataFrames in IPython or terminal tables. False
        suppresses display but still returns the frames and numerical values.

    Returns
    -------
    dict[str, object]
        Mode count, d, N_t, memory estimates in GiB, sparse-entry upper bound,
        heuristic feasible flag, grid descriptions, and pandas DataFrames
        ``grid_table``/``resource_table``. Existing scalar/dictionary keys remain
        available. Grid reports show effective IR/UV and equivalent explicit M.

    Notes
    -----
    One complex128 ket costs 16*d bytes. history_gib counts retained solver
    kets; retained_vectors_gib approximates twice that for copied coefficients.
    The upper bound nnz <= d*(1+2*M) includes the diagonal. Feasibility means
    d<=100000 and retained_vectors_gib<=1; it is a heuristic, not a guarantee.
    Sparse matrices, basis/Python objects, solver workspace and plots are
    excluded (also recorded in resource_table.attrs). No basis is enumerated.
    """
    config = ExperimentConfig(param_photon or {}, param_atom, param_time_evol, cutoffs,
                              n_max, truncation, store_state=store_state,
                              CTRL_M_EXPLICIT=CTRL_M_EXPLICIT, M=M,
                              mode_selection=mode_selection, photon_window=photon_window,
                              atom_window=atom_window)
    base, selected = resolve_grid(config)
    modes, cap = len(selected), integer(n_max, "n_max")
    if truncation == "full+totalcap":
        dimension = 2 * comb(modes + cap, cap)
    elif truncation == "truncated":
        dimension = 2 * (1 + modes * cap)
    elif truncation == "full":
        dimension = 2 * (cap + 1) ** modes
    else:
        raise ValueError(f"Unknown truncation: {truncation}")
    times = output_times(param_time_evol)
    ket_gib = dimension * 16 / 2**30
    history_gib = ket_gib * (len(times) if store_state else 1)
    retained_gib = 2 * history_gib
    nnz_bound = dimension * (1 + 2 * modes)
    summary = grid_summary(base, selected, param_atom["L"])
    feasible = dimension <= 100_000 and retained_gib <= 1.0
    estimate = {"n_modes": modes, "dimension": dimension, "n_times": len(times),
                "ket_gib": ket_gib, "history_gib": history_gib,
                "retained_vectors_gib": retained_gib, "hamiltonian_nnz_bound": nnz_bound,
                "feasible": feasible, "grid": summary}
    estimate.update(resource_tables(estimate, config))
    if print_report:
        display_resource_tables(estimate)
    return estimate


def estimate_config(config, print_report=True):
    """Estimate resources using the exact inputs of a shared configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Grid, selection, photon cap, basis, output times and storage policy.
    print_report : bool, optional
        Display tabular estimates when True; False returns them silently.

    Returns
    -------
    dict[str, object]
        resource_estimation output, including numerical entries and reusable
        ``grid_table``/``resource_table`` pandas DataFrames. No propagation occurs.
    """
    return resource_estimation(config.param_atom, config.param_time_evol, config.cutoffs,
                               config.n_max, config.truncation, config.store_state,
                               config.CTRL_M_EXPLICIT, config.M, config.param_photon,
                               config.mode_selection, config.photon_window, config.atom_window,
                               print_report)


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
        self.result = qutip.sesolve(self.H, self.state0, self.times, options={
            "method": self.param_time_evol.get("method", "bdf"),
            "store_states": self.store_state, "store_final_state": True,
            "normalize_output": False, "rtol": self.param_time_evol.get("rtol", 1e-9),
            "atol": self.param_time_evol.get("atol", 1e-11),
            "progress_bar": "tqdm" if progress else False})
        vectors = np.array([s.full()[:, 0] for s in self.result.states]) if self.store_state \
            else self.result.final_state.full()[:, 0]
        self.c_g_array, self.c_e_array = vectors[..., 0::2], vectors[..., 1::2]
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
        if self.c_g_array is None:
            raise RuntimeError("Call propagate_state first")
        ground, excited = np.atleast_2d(self.c_g_array), np.atleast_2d(self.c_e_array)
        probability = np.abs(ground) ** 2 + np.abs(excited) ** 2
        indices, modes = self._one_photon_indices()
        self.observables = {"time": self.times if self.store_state else self.times[-1:],
                            "norm": probability.sum(axis=1),
                            "p_excited": (np.abs(excited) ** 2).sum(axis=1)}
        for name, mask in (("p_transmitted", self.k_tab[modes] > 0),
                           ("p_reflected", self.k_tab[modes] < 0),
                           ("p_zero_1g", self.k_tab[modes] == 0)):
            self.observables[name] = (np.abs(ground[:, indices[mask]]) ** 2).sum(axis=1)
        for n in range(int(self.photon_numbers.max()) + 1):
            self.observables[f"p_{n}_photon"] = probability[:, self.photon_numbers == n].sum(axis=1)
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
        return observables_table(self.observables, summary, self.store_state)

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

    def compute_entropy(self, t=None):
        """Evaluate the normalized TLS von Neumann entropy in natural logarithms.

        Parameters
        ----------
        t : float or None, optional
            Time selector: None for all retained outputs, -1 for final, or nearest
            stored output to an in-range time; follows _vectors_for.

        Returns
        -------
        numpy.ndarray or float
            Real array (K,) for None or scalar otherwise. With
            rho_hat=Tr_field(|psi><psi|)/||psi||**2, S=-Tr(rho_hat*log(rho_hat))
            lies in [0,log(2)]. For the pure joint state this is field/TLS
            entanglement entropy. Normalization here removes solver norm drift.
        """
        values = np.array([qutip.entropy_vn(qutip.Qobj(atom_density_matrix(v) / np.vdot(v, v).real))
                           for v in self._vectors_for(t)])
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
