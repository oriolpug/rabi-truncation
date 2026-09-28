"""Pure-state dynamics and diagnostics shared by every experiment."""

from math import comb

import numpy as np
import qutip

from .energy_profile import EnergyProfile
from .fidelities import atom_density_matrix
from .grid import grid_summary, integer, momentum_modes, resolve_grid
from .hamiltonians import Hamiltonian
from .states import FockBasis, initial_state
from .xp_config import ExperimentConfig


def output_times(param_time_evol):
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
    if print_report:
        control = f"explicit M={M}" if CTRL_M_EXPLICIT else f"cutoffs={cutoffs}"
        print(f"Grid control: {control}; L={param_atom['L']:g}; delta_k={summary['delta_k']:.8g}")
        for label in ("base", "selected"):
            item = summary[label]
            print(f"  {label}: M={item['n_modes']}; k in [{item['k_min']:.8g}, {item['k_max']:.8g}]; "
                  f"effective (IR,UV)=({item['ir_effective']:.8g},{item['uv_effective']:.8g}); "
                  f"zero={item['zero_present']}")
            print(f"    radial cutoffs exact={item['radial_cutoffs_exact']}; "
                  f"equivalent explicit M={item['equivalent_M']}")
        print(f"Basis: {truncation}; dimension={dimension:,}; output times={len(times)}")
        print(f"Ket={ket_gib:.4g} GiB; solver history={history_gib:.4g} GiB; "
              f"retained vectors approximately={retained_gib:.4g} GiB")
        print(f"Hamiltonian nnz <= {nnz_bound:,}; heuristic feasible={feasible}")
        print("Memory excludes sparse matrices, basis/Python objects, solver workspace and plots.")
    return estimate


def estimate_config(config, print_report=True):
    return resource_estimation(config.param_atom, config.param_time_evol, config.cutoffs,
                               config.n_max, config.truncation, config.store_state,
                               config.CTRL_M_EXPLICIT, config.M, config.param_photon,
                               config.mode_selection, config.photon_window, config.atom_window,
                               print_report)


class Experiment:
    def __init__(self, config):
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
        indices = np.flatnonzero(self.photon_numbers == 1)
        mode_ids = np.array([np.argmax(self.basis.states[i]) for i in indices], dtype=int)
        return indices, mode_ids

    def compute_observables(self):
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

    def one_photon_wavefunction(self, i, x_tab):
        if self.c_g_array is None:
            raise RuntimeError("Call propagate_state first")
        coefficients = self.c_g_array[i] if self.store_state else self.c_g_array
        indices, modes = self._one_photon_indices()
        return np.exp(1j * np.outer(x_tab, self.k_tab[modes])) @ coefficients[indices] / \
            np.sqrt(self.param_atom["L"])

    def _vectors_for(self, t=None):
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
        matrices = [atom_density_matrix(v) for v in self._vectors_for(t)]
        return matrices if t is None else matrices[0]

    def compute_excited_probability(self, t=None):
        values = np.array([atom_density_matrix(v)[1, 1].real for v in self._vectors_for(t)])
        return values if t is None else float(values[0])

    def compute_entropy(self, t=None):
        values = np.array([qutip.entropy_vn(qutip.Qobj(atom_density_matrix(v) / np.vdot(v, v).real))
                           for v in self._vectors_for(t)])
        return values if t is None else float(values[0])

    def compute_energy(self, t=None):
        matrix = self.hamiltonian.H0 + self.param_atom["D"] * self.hamiltonian.V
        values = np.array([np.vdot(v, matrix @ v).real for v in self._vectors_for(t)])
        return values if t is None else float(values[0])

    def compute_energy_profile_modes(self, t=None):
        profile = EnergyProfile(self)
        results = [profile.energy_modes_vec(v) for v in self._vectors_for(t)]
        modes, atom = np.array([r[0] for r in results]), np.array([r[1] for r in results])
        return self.k_tab, modes if t is None else modes[0], 0.0, atom if t is None else float(atom[0])

    def compute_energy_profile_excitations(self, t=None):
        profile = EnergyProfile(self)
        values = np.array([profile.energy_excitations_vec(v) for v in self._vectors_for(t)])
        return profile.excitation_axis, values if t is None else values[0]
