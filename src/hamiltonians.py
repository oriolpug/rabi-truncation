"""
Implements the Quantum Rabi Hamiltonian in its different setups
config -> dictionary with relevant parameters/assumptions of the simulation
Hamiltonian -> abstract class implementing common utilities
Precise implementations in child classes
HamiltonianFull: only excitation cap
HamiltonianTruncated: original truncation scheme, no cross-mode excitation states
HamiltonianBand: HamiltonianTruncated with an auxiliary state with the same frequency as the atom
"""

import qutip
import numpy as np
import scipy.sparse as sp
from utilities import Config
from abc import ABC, abstractmethod
from itertools import product
from states import FullBasis, TruncatedBasis, AtomBasis

class Hamiltonian(ABC):
    def __init__(self, config):
        self.config = config
        self.d = self.compute_dim() # implemented in child classes
        self.H = sp.lil_matrix((self.d,self.d), dtype=complex)

    # Abstract methods: implemented depending on precise hamiltonian
    @abstractmethod
    def compute_dim(self) -> int: ...

    @abstractmethod
    def free(self) -> sp.lil_matrix: ...

    @abstractmethod
    def interaction(self) -> sp.lil_matrix: ...

    # Calculate indices {'n1': 1, 'n3': 4, 'atom': 'g'} -> integer index mapping
    @abstractmethod
    def state_to_index(self, state: dict) -> int: ...

    # get item: val = H[{state_1}, {state_2}]
    def __getitem__(self, states: tuple[dict, dict]) -> complex:
        row, col = states
        return self.H[self.state_to_index(row), self.state_to_index(col)]

    # set item: H[{state_1}, {state_2}] = val
    def __setitem__(self, states: tuple[dict, dict], value: complex):
        row, col = states
        self.H[self.state_to_index(row), self.state_to_index(col)] = value

    def build_hamiltonian(self):
        self.free()
        self.interaction()

    def g(self, k) -> complex:
        return 1j * self.config.g * np.sqrt(np.abs(k))

    def atom_index(self, atom):
        return 1 if atom == "e" else 0

    def to_qObj(self):
        return qutip.Qobj(self.H)

    def transition_possible(self, ket, bra) -> bool:
        photon_diffs =  [
          ket.get(f'n{m+1}', 0) - bra.get(f'n{m+1}', 0)
          for m in range(self.config.modes)
        ]
        nonzero_diffs = [d for d in photon_diffs if d != 0]
        atom_change = self.atom_index(ket['atom']) - self.atom_index(bra['atom'])

        if atom_change == 0 or len(nonzero_diffs) != 1 :
            return False
        else:
            if self.config.RWA:
                if atom_change + nonzero_diffs[0] == 0: return True
                else:
                    return False
            else: # no RWA
                return abs(nonzero_diffs[0]) == 1

    def transition_sign(self, ket, bra) -> int:
        # Atom gains -> +, atom loses -> -
        photon_diffs =  [
          ket.get(f'n{m+1}', 0) - bra.get(f'n{m+1}', 0)
          for m in range(self.config.modes)
        ]
        nonzero_diffs = [d for d in photon_diffs if d != 0]
        assert len(nonzero_diffs) == 1 and abs(nonzero_diffs[0]) == 1, f"Invalid transition {bra} -> {ket}"
        return nonzero_diffs[0]

    def transition_location(self, ket, bra):
        photon_diffs = [
            ket.get(f'n{m + 1}', 0) - bra.get(f'n{m + 1}', 0)
            for m in range(self.config.modes)
        ]
        nonzero_diffs = [(i,d) for i, d in enumerate(photon_diffs) if d != 0]
        return nonzero_diffs[0][0]

class HamiltonianFull(FullBasis, Hamiltonian):
    def free(self):
        M = self.config.modes
        hbar, c = self.config.hbar, self.config.c
        w, ks = self.config.w_atom, self.config.frequencies

        # Free H: count excitations in each mode/atom and add the corresponding energy
        for state in self.all_states():
            if state['atom'] == 'g':
                self[state, state] = hbar * c * np.sum( [state.get(f'n{m+1}',0) * np.abs(ks[m]) for m in range(M)])
            elif state['atom'] == 'e':
                self[state, state] = hbar * ( w + c * np.sum( [state.get(f'n{m+1}',0) * np.abs(ks[m]) for m in range(M)]) )

    def interaction(self):
        hbar, c = self.config.hbar, self.config.c
        ks, x = self.config.frequencies, self.config.x_atom

        # Interaction H: energy for each pair of states that makes a feasible transition (otherwise zero)
        for ket, bra in product(self.all_states(), repeat=2):
            if self.transition_possible(ket, bra):
                sign = self.transition_sign(ket, bra)
                trans_index = self.transition_location(ket, bra)
                eigenvalue = np.sqrt(max(bra.get(f'n{trans_index+1}',0),
                                         ket.get(f'n{trans_index+1}', 0))) # eigenvalue of the bosonic operator

                self[ket, bra] = hbar * sign * self.g(ks[trans_index]) * eigenvalue * np.exp(sign * 1j * ks[trans_index] * x)

class HamiltonianTruncated(TruncatedBasis, Hamiltonian):
    # state: |n_m ; s>

    def free(self):
        hbar, c = self.config.hbar, self.config.c
        w, ks = self.config.w_atom, self.config.frequencies

        for state in self.all_states():
            key = next(k for k in state if k != 'atom')
            m = int(key[1:]) - 1
            n = state[key]
            atom = state.get('atom', 0)

            self[state, state] = hbar * c * n * np.abs(ks[m]) + self.atom_index(atom) * w

    def interaction(self):
        hbar, c = self.config.hbar, self.config.c
        w, ks, x = self.config.w_atom, self.config.frequencies, self.config.x_atom

        for ket, bra in product(self.all_states(), repeat=2):
            if self.transition_possible(ket, bra):
                sign = self.transition_sign(ket, bra)
                trans_index = self.transition_location(ket, bra)
                eigenvalue = np.sqrt(max(bra.get(f'n{trans_index+1}', 0),
                                         ket.get(f'n{trans_index+1}', 0)))  # eigenvalue of the bosonic operator

                self[ket, bra] = hbar * sign * self.g(ks[trans_index]) * eigenvalue * np.exp(sign * 1j * ks[trans_index] * x)


class HamiltonianAtom(AtomBasis, Hamiltonian):
    # state: |n_m, n_atom ; s>

    def free(self):
        hbar, c = self.config.hbar, self.config.c
        w, ks = self.config.w_atom, self.config.frequencies

        for state in self.all_states():
            key = next(k for k in state if k not in ['atom', 'n_atom'])
            m = int(key[1:]) - 1
            n = state[key]
            n_atom = state.get("n_atom", 0)
            atom = state.get('atom', 0)

            self[state, state] = hbar * c * n * np.abs(ks[m]) + hbar * n_atom * np.abs(w) + self.atom_index(atom) * w

    def transition_sign(self, ket, bra) -> int:
        # Atom gains -> +, atom loses -> -
        photon_diffs = [
          ket.get(f'n{m+1}', 0) - bra.get(f'n{m+1}', 0)
          for m in range(self.config.modes)
        ]
        photon_diffs.append(ket.get('n_atom', 0) - bra.get('n_atom', 0))

        nonzero_diffs = [d for d in photon_diffs if d != 0]
        assert len(nonzero_diffs) == 1 and abs(nonzero_diffs[0]) == 1, f"Invalid transition {bra} -> {ket}"
        return nonzero_diffs[0]

    def transition_location(self, ket, bra):
        photon_diffs = [
            ket.get(f'n{m + 1}', 0) - bra.get(f'n{m + 1}', 0)
            for m in range(self.config.modes)
        ]
        photon_diffs.append(ket.get('n_atom', 0) - bra.get('n_atom', 0))
        nonzero_diffs = [(i,d) for i, d in enumerate(photon_diffs) if d != 0]
        return nonzero_diffs[0][0]

    def transition_possible(self, ket, bra) -> bool:
        photon_diffs = [
          ket.get(f'n{m+1}', 0) - bra.get(f'n{m+1}', 0)
          for m in range(self.config.modes)
        ]
        photon_diffs.append(ket.get('n_atom', 0) - bra.get('n_atom', 0))

        nonzero_diffs = [d for d in photon_diffs if d != 0]
        atom_change = self.atom_index(ket['atom']) - self.atom_index(bra['atom'])

        if atom_change == 0 or len(nonzero_diffs) != 1 :
            return False
        else:
            if self.config.RWA:
                if atom_change + nonzero_diffs[0] == 0: return True
                else:
                    return False
            else: # no RWA
                return abs(nonzero_diffs[0]) == 1

    def interaction(self):
        hbar, c = self.config.hbar, self.config.c
        w, ks, x = self.config.w_atom, self.config.frequencies, self.config.x_atom

        for ket, bra in product(self.all_states(), repeat=2):
            if self.transition_possible(ket, bra):
                sign = self.transition_sign(ket, bra)
                trans_index = self.transition_location(ket, bra)
                eigenvalue = np.sqrt(max(bra.get(f'n{trans_index+1}', 0),
                                         ket.get(f'n{trans_index+1}', 0)))  # eigenvalue of the bosonic operator
                k_trans = ks[trans_index] if trans_index < len(ks) else w

                self[ket, bra] = hbar * sign * self.g(k_trans) * eigenvalue * np.exp(
                    sign * 1j * k_trans * x)

def hamiltonian(config: Config) -> qutip.Qobj:
    # Logic to choose appropriate Hamiltonian from config

    match config.truncation:
        case "truncated":
            h = HamiltonianTruncated(config)
        case "truncated+atom":
            h = HamiltonianAtom(config)
        case "full":
            h = HamiltonianFull(config)

    h.build_hamiltonian()
    return h.to_qObj()