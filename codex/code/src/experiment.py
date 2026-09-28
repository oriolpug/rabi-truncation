"""A notebook-friendly scattering experiment with QuTiP propagation."""

from math import comb, pi

import numpy as np
import qutip

from src.hamiltonians import Hamiltonian
from src.states import FockBasis, initial_state


def momentum_modes(param_atom, cutoffs):
    """Signed periodic-box momenta in the chosen IR/UV band."""
    if "k_modes" in cutoffs:
        k_tab = np.asarray(cutoffs["k_modes"], dtype=float)
        if k_tab.ndim != 1 or len(k_tab) == 0 or len(np.unique(k_tab)) != len(k_tab):
            raise ValueError("k_modes must be a nonempty list of distinct wavevectors")
        return np.sort(k_tab)

    L = param_atom["L"]
    m_max = int(np.ceil(cutoffs["uv_cutoff"] * L / (2 * pi)))
    k_tab = 2 * pi * np.arange(-m_max, m_max + 1) / L
    band = (np.abs(k_tab) >= cutoffs["ir_cutoff"] - 1e-12) & \
           (np.abs(k_tab) <= cutoffs["uv_cutoff"] + 1e-12)
    return k_tab[band]


def resource_estimation(param_atom, param_time_evol, cutoffs,
                        n_max=3, truncation="full+totalcap", store_state=True):
    
    """Count the physical basis and stored ket history before allocating them."""

    M = len(momentum_modes(param_atom, cutoffs))
    N = n_max
    if M == 0 or N < 1:
        raise ValueError("At least one mode and n_max >= 1 are required")
    dimensions = {
        "full+totalcap": 2 * comb(M + N, N),
        "truncated": 2 * (1 + M * N),
        "full": 2 * (N + 1) ** M,
    }
    if truncation not in dimensions:
        raise ValueError(f"Unknown truncation: {truncation}")
    
    T, dt = param_time_evol["T"], param_time_evol["dt"]
    if T <= 0 or dt <= 0:
        raise ValueError("T and dt must be positive")
    
    n_times = int(np.floor(T / dt)) + 1
    dim = dimensions[truncation]
    history_gib = dim * 16 * (n_times if store_state else 1) / 2**30
    feasible = dim <= 100_000 and history_gib <= 1.0
    print(f"Modes: {M}; truncation: {truncation}; ket dimension: {dim:,}")
    print(f"Output times: {n_times}; ket/history: {history_gib:.3g} GiB")
    print(f"Feasible: {feasible} (excludes Hamiltonian and Python objects)")
    return {"n_modes": M, "dimension": dim, "n_times": n_times,
            "history_gib": history_gib, "feasible": feasible}


class Experiment:
    """One Gaussian packet scattered by a TLS on a chosen Fock basis."""

    def __init__(self, config):
        self.param_photon = config.param_photon
        self.param_atom = config.param_atom
        self.param_time_evol = config.param_time_evol
        self.n_max = config.n_max
        self.truncation = config.truncation
        self.RWA = config.RWA
        self.store_state = config.store_state

        self.k_tab = momentum_modes(self.param_atom, config.cutoffs)
        self.n_modes = len(self.k_tab)

        if self.n_modes == 0:
            raise ValueError("The momentum band contains no modes")
        
        self.basis = FockBasis(self.n_modes, self.n_max, self.truncation)
        self.photon_numbers = self.basis.photon_numbers
        self.hamiltonian = Hamiltonian(self.basis, self.k_tab, self.param_atom, self.RWA)
        self.H = self.hamiltonian.build_hamiltonian()
        self.state0 = initial_state(self.basis, self.k_tab, self.param_photon, self.param_atom)
        self.c_g_array = None
        self.c_e_array = None
        self.observables = None

    def propagate_state(self, progress=False):
        """Evolve the pure state; dt specifies output spacing, not solver step size."""
        T, dt = self.param_time_evol["T"], self.param_time_evol["dt"]
        if T <= 0 or dt <= 0:
            raise ValueError("T and dt must be positive")
        self.times = np.arange(int(np.floor(T / dt)) + 1) * dt
        method = self.param_time_evol.get("method", "bdf")
        self.result = qutip.sesolve(
            self.H, self.state0, self.times,
            options={"method": method, 
                     "store_states": self.store_state,
                     "store_final_state": True, 
                     "normalize_output": False,
                     "rtol": self.param_time_evol.get("rtol", 1e-9),
                     "atol": self.param_time_evol.get("atol", 1e-11),
                     "progress_bar": "tqdm" if progress else False},
        )
        if self.store_state:
            vectors = np.array([state.full()[:, 0] for state in self.result.states])
            self.c_g_array, self.c_e_array = vectors[:, 0::2], vectors[:, 1::2]
        else:
            vector = self.result.final_state.full()[:, 0]
            self.c_g_array, self.c_e_array = vector[0::2], vector[1::2]
        return self.c_g_array, self.c_e_array

    def _one_photon_indices(self):
        """Indices of field states containing exactly one photon."""
        return np.array([
            i for i, _ in enumerate(self.basis.states)
            if self.photon_numbers[i] == 1
        ], dtype=int)

    def compute_observables(self):
        if self.c_g_array is None:
            raise RuntimeError("Call propagate_state first")
        c_g = np.atleast_2d(self.c_g_array)
        c_e = np.atleast_2d(self.c_e_array)
        probability = np.abs(c_g) ** 2 + np.abs(c_e) ** 2
        one_photon = self._one_photon_indices()
        mode_index = np.array([np.argmax(self.basis.states[i][:self.n_modes])
                               for i in one_photon], dtype=int)
        right = one_photon[self.k_tab[mode_index] > 0]
        left = one_photon[self.k_tab[mode_index] < 0]

        self.observables = {
            "time": self.times if self.store_state else self.times[-1:],
            "norm": probability.sum(axis=1),
            "p_excited": (np.abs(c_e) ** 2).sum(axis=1),
            "p_transmitted": (np.abs(c_g[:, right]) ** 2).sum(axis=1),
            "p_reflected": (np.abs(c_g[:, left]) ** 2).sum(axis=1),
        }
        for n in range(int(self.photon_numbers.max()) + 1):
            self.observables[f"p_{n}_photon"] = probability[:, self.photon_numbers == n].sum(axis=1)
        return self.observables

    def one_photon_wavefunction(self, i, x_tab):
        """Schrodinger-picture wavefunction of the |one photon; g> sector."""
        if self.c_g_array is None:
            raise RuntimeError("Call propagate_state first")
        c_g = self.c_g_array[i] if self.store_state else self.c_g_array
        one_photon = self._one_photon_indices()
        mode_index = np.array([np.argmax(self.basis.states[j][:self.n_modes])
                               for j in one_photon], dtype=int)
        x_tab = np.asarray(x_tab)
        return np.exp(1j * np.outer(x_tab, self.k_tab[mode_index])) @ \
               c_g[one_photon] / np.sqrt(self.param_atom["L"])
