"""
State vector classes for each Hamiltonian type, mirroring hamiltonians.py.

Basis mixins (FullBasis, TruncatedBasis, AtomBasis) are the single source of
truth for compute_dim and state_to_index; they are shared with hamiltonians.py.

Each State class can be initialised as a NumberState or CoherentState (from
utilities), and takes a vector ck (len=modes, norm=1) of per-mode coefficients,
allowing Gaussian wavepacket initialisation as a superposition over modes.
"""

import numpy as np
from math import factorial
from abc import ABC, abstractmethod
from itertools import product as iterproduct
from typing import Optional

import qutip
from utilities import Config, NumberState, CoherentState, StateType


def _coherent_coeff(alpha: complex, n: int) -> complex:
    """Fock coefficient <n|alpha> = e^{-|alpha|²/2} * alpha^n / sqrt(n!)"""
    return np.exp(-abs(alpha) ** 2 / 2) * alpha ** n / np.sqrt(factorial(n))


# ---------------------------------------------------------------------------
# Basis mixins: single source of truth for dimension and indexing
# Python mixin behaviour: self refers to concrete instance
# ---------------------------------------------------------------------------

class FullBasis:
    """Basis for HamiltonianFull / StateFull: unconstrained Fock space up to N."""

    def __new__(cls, *args, **kwargs):
        if cls is FullBasis:
            raise TypeError("FullBasis is a mixin and cannot be instantiated directly")
        return super().__new__(cls)

    def compute_dim(self) -> int:
        return 2 * (self.config.excitation_cap + 1) ** self.config.modes

    def state_to_index(self, state: dict) -> int:
        M, N = self.config.modes, self.config.excitation_cap
        idx = 0
        for m in range(M):
            idx = idx * (N + 1) + state.get(f'n{m+1}', 0)
        return idx * 2 + self.atom_index(state.get('atom', 'g'))

    def all_states(self):
        photon_range = range(self.config.excitation_cap + 1)
        for *ns, atom in iterproduct(*[photon_range] * self.config.modes, ['g', 'e']):
            yield {f'n{m + 1}': ns[m] for m in range(self.config.modes)} | {'atom': atom}

class TruncatedBasis:
    """Basis for HamiltonianTruncated / StateTruncated: vacuum + single-mode excitations."""

    def __new__(cls, *args, **kwargs):
        if cls is TruncatedBasis:
            raise TypeError("TruncatedBasis is a mixin and cannot be instantiated directly")
        return super().__new__(cls)

    def compute_dim(self) -> int:
        return 2 * (self.config.modes * self.config.excitation_cap + 1)

    def state_to_index(self, state: dict) -> int:
        N, M = self.config.excitation_cap, self.config.modes
        idx = 0
        for m in range(M):
            n = state.get(f'n{m+1}', 0)
            if n != 0:
                idx = N * m + n
                break
        return idx * 2 + self.atom_index(state.get('atom', 'g'))

    def all_states(self):
        N = self.config.excitation_cap
        M = self.config.modes

        for m in range(M):
            for n in range(N + 1):
                for atom in ['g', 'e']:
                    yield {f'n{m+1}': n, 'atom': atom}

class AtomBasis:
    """Basis for HamiltonianAtom / StateAtom: TruncatedBasis extended with n_atom oscillator."""

    def __new__(cls, *args, **kwargs):
        if cls is AtomBasis:
            raise TypeError("AtomBasis is a mixin and cannot be instantiated directly")
        return super().__new__(cls)

    def compute_dim(self) -> int:
        N, M = self.config.excitation_cap, self.config.modes
        return 2 * (M * N + 1) * (N + 1)

    def state_to_index(self, state: dict) -> int:
        N, M = self.config.excitation_cap, self.config.modes
        n_atom = state.get('n_atom', 0)
        idx = n_atom  # vacuum photon sector: indices 0..N
        for m in range(M):
            n = state.get(f'n{m+1}', 0)
            if n != 0:
                idx = (N + 1) + m * N * (N + 1) + (n - 1) * (N + 1) + n_atom
                break
        return idx * 2 + self.atom_index(state.get('atom', 'g'))

    def all_states(self):
        N = self.config.excitation_cap
        M = self.config.modes

        for m in range(M):
            for n in range(N + 1):
                for n_atom in range(N + 1):
                    for atom in ['g', 'e']:
                        yield {f'n{m+1}': n, 'n_atom': n_atom, 'atom': atom}
# ---------------------------------------------------------------------------
# State base and concrete classes
# ---------------------------------------------------------------------------

class State(ABC):
    def __init__(self, config: Config, state_type: StateType):
        self.config = config
        self.state_type = state_type
        self.ck = self.calculate_ck(config)
        self.v = np.zeros(self.compute_dim(), dtype=complex)
        self.build_vector()

    @staticmethod
    def calculate_ck(config) -> list[complex]:
        ck = [np.exp(- 0.5 * config.sigma_photon ** 2 * (config.frequencies[n] - config.k_photon) ** 2) * np.exp(- 1j * config.frequencies[n] * config.x_photon)
              for n in range(config.modes)]
        return ck / np.linalg.norm(ck)

    @abstractmethod
    def compute_dim(self) -> int: ...

    @abstractmethod
    def state_to_index(self, state: dict) -> int: ...

    @abstractmethod
    def build_vector(self): ...

    def atom_index(self, atom: str) -> int:
        return 1 if atom == "e" else 0

    @classmethod
    def from_vector(cls, config: Config, v: np.ndarray) -> 'State':
        """Initialise directly from a pre-computed coefficient vector.

        Bypasses state_type / calculate_ck / build_vector entirely.
        The vector is normalised before storage.
        """
        obj = object.__new__(cls)
        obj.config = config
        obj.state_type = None
        obj.ck = None
        obj.v = np.array(v, dtype=complex)
        obj.v /= np.linalg.norm(obj.v)
        return obj

    def to_qobj(self) -> qutip.Qobj:
        return qutip.Qobj(self.v)

    def photon_density_matrix(self) -> np.ndarray:
        """Reduced density matrix of the photon field, tracing over the atom.

        ρ_photon[k, k'] = Σ_s c_{k,s} * conj(c_{k',s})
                        = outer(v_g, v_g*) + outer(v_e, v_e*)
        Result is (dim//2 x dim//2), with photon states indexed as in state_to_index // 2.
        """
        v_g = self.v[0::2]
        v_e = self.v[1::2]
        return np.outer(v_g, v_g.conj()) + np.outer(v_e, v_e.conj())

    def atom_density_matrix(self) -> np.ndarray:
        """2x2 reduced density matrix of the atom, tracing over photon degrees of freedom.

        All bases store atom as the lowest index bit (idx*2 + atom_index), so
        v[0::2] = ground amplitudes, v[1::2] = excited amplitudes for any basis.
        Rows/cols ordered as [g, e].
        """
        v_g = self.v[0::2]
        v_e = self.v[1::2]
        return np.array([
            [v_g @ v_g.conj(), v_g @ v_e.conj()],
            [v_e @ v_g.conj(), v_e @ v_e.conj()]
        ])

    # get item: val = H[{state_1}, {state_2}]
    def __getitem__(self, s: dict) -> complex:
        return self.v[self.state_to_index(s)]

    # set item: H[{state_1}, {state_2}] = val
    def __setitem__(self, s: dict, value: complex):
        self.v[self.state_to_index(s)] = value


class StateFull(FullBasis, State):
    def build_vector(self):
        M, N = self.config.modes, self.config.excitation_cap
        atom_coeffs = self.config.atom_coeffs

        if isinstance(self.state_type, NumberState):
            n = self.state_type.number
            for m in range(M):
                base = {f'n{mm+1}': (n if mm == m else 0) for mm in range(M)}
                for atom, a_coeff in atom_coeffs.items():
                    self.v[self.state_to_index(base | {'atom': atom})] += self.ck[m] * a_coeff

        elif isinstance(self.state_type, CoherentState):
            # Product coherent state: (x)_m |alpha * ck[m]>  x  |atom>
            alpha = self.state_type.alpha
            for *ns, atom in iterproduct(*[range(N + 1)] * M, ['g', 'e']):
                a_coeff = atom_coeffs.get(atom, 0)
                if a_coeff == 0:
                    continue
                coeff = a_coeff * np.prod([_coherent_coeff(alpha * self.ck[m], ns[m]) for m in range(M)])
                state = {f'n{m+1}': ns[m] for m in range(M)} | {'atom': atom}
                self.v[self.state_to_index(state)] += coeff

        self.v /= np.linalg.norm(self.v)

class StateTruncated(TruncatedBasis, State):
    def build_vector(self):
        N, M = self.config.excitation_cap, self.config.modes
        atom_coeffs = self.config.atom_coeffs

        if isinstance(self.state_type, NumberState):
            n = self.state_type.number
            for m in range(M):
                for atom, a_coeff in atom_coeffs.items():
                    state = {f'n{m+1}': n, 'atom': atom}
                    self.v[self.state_to_index(state)] += self.ck[m] * a_coeff

        elif isinstance(self.state_type, CoherentState):
            # Project product coherent state onto truncated subspace:
            #   <0|psi>   = exp(-|alpha|^2 / 2)
            #   <n_m|psi> = (alpha*ck[m])^n / sqrt(n!) * exp(-|alpha|^2 / 2)
            alpha = self.state_type.alpha
            vac_amp = np.exp(-abs(alpha) ** 2 / 2)
            for atom, a_coeff in atom_coeffs.items():
                if a_coeff == 0:
                    continue
                self.v[self.state_to_index({'atom': atom})] += a_coeff * vac_amp
                for m in range(M):
                    alpha_m = alpha * self.ck[m]
                    for n in range(1, N + 1):
                        coeff = alpha_m ** n / np.sqrt(factorial(n)) * vac_amp
                        state = {f'n{m+1}': n, 'atom': atom}
                        self.v[self.state_to_index(state)] += a_coeff * coeff

        self.v /= np.linalg.norm(self.v)

class StateAtom(AtomBasis, State):
    def build_vector(self):
        N, M = self.config.excitation_cap, self.config.modes
        atom_coeffs = self.config.atom_coeffs

        if isinstance(self.state_type, NumberState):
            n = self.state_type.number
            for m in range(M):
                for atom, a_coeff in atom_coeffs.items():
                    state = {f'n{m+1}': n, 'n_atom': 0, 'atom': atom}
                    self.v[self.state_to_index(state)] += self.ck[m] * a_coeff

        elif isinstance(self.state_type, CoherentState):
            # Same projection as StateTruncated; n_atom initialised in vacuum
            alpha = self.state_type.alpha
            vac_amp = np.exp(-abs(alpha) ** 2 / 2)
            for atom, a_coeff in atom_coeffs.items():
                if a_coeff == 0:
                    continue
                self.v[self.state_to_index({'n_atom': 0, 'atom': atom})] += a_coeff * vac_amp
                for m in range(M):
                    alpha_m = alpha * self.ck[m]
                    for n in range(1, N + 1):
                        coeff = alpha_m ** n / np.sqrt(factorial(n)) * vac_amp
                        state = {f'n{m+1}': n, 'n_atom': 0, 'atom': atom}
                        self.v[self.state_to_index(state)] += a_coeff * coeff

            self.v /= np.linalg.norm(self.v)

def state(config: Config, v: Optional[np.ndarray] = None) -> qutip.Qobj:
    cls = {"truncated": StateTruncated, "truncated+atom": StateAtom, "full": StateFull}[config.truncation]
    if v is not None:
        return cls.from_vector(config, v).to_qobj()
    return cls(config, config.state).to_qobj()
