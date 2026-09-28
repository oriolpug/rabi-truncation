"""Fock bases and initial states. The TLS is the last, two-valued index."""

from itertools import combinations_with_replacement, product
from math import factorial

import numpy as np
import qutip


class FockBasis:
    def __init__(self, modes, cap, truncation):
        self.modes = modes
        self.cap = cap
        self.truncation = truncation
        vacuum = (0,) * modes
        if truncation == "truncated":
            field_states = [vacuum]
            for m in range(modes):
                for n in range(1, cap + 1):
                    occupation = list(vacuum)
                    occupation[m] = n
                    field_states.append(tuple(occupation))
        elif truncation == "full+totalcap":
            field_states = []
            for total in range(cap + 1):
                for occupied_modes in combinations_with_replacement(range(modes), total):
                    field_states.append(tuple(np.bincount(occupied_modes, minlength=modes)))
        elif truncation == "full":
            field_states = list(product(range(cap + 1), repeat=modes))
        else:
            raise ValueError(f"Unknown truncation: {truncation}")

        self.states = field_states
        self.index = {occupation: i for i, occupation in enumerate(self.states)}
        self.photon_numbers = np.array([sum(n) for n in self.states])
        self.dim = 2 * len(self.states)


def initial_state(basis, k_tab, param_photon, param_atom):
    """One incoming photon by default; optional number/coherent preparations."""
    k0 = param_photon["k_0"]
    sigma = param_photon["sigma_k"]
    if k0 <= 0 or sigma <= 0:
        raise ValueError("k_0 and sigma_k must be positive")

    packet = np.exp(-(k_tab - k0) ** 2 / (4 * sigma ** 2))
    packet = packet * np.exp(-1j * k_tab * param_photon["x_0"])
    if param_photon.get("right_moving_only", True):
        packet[k_tab <= 0] = 0
    packet_norm = np.linalg.norm(packet)
    if packet_norm == 0:
        raise ValueError("The selected modes contain no incident packet")
    packet /= packet_norm

    atom = param_atom.get("initial_state", "g")
    atom_coeffs = {
        "g": (1, 0), 
        "e": (0, 1),
        "+": (1 / np.sqrt(2), 1 / np.sqrt(2)),
        "-": (1 / np.sqrt(2), -1 / np.sqrt(2)),
    }
    if atom not in atom_coeffs:
        raise ValueError("initial_state must be g, e, +, or -")
    atom_vector = np.asarray(atom_coeffs[atom], dtype=complex)

    kind = param_photon.get("state", "number")
    v = np.zeros(basis.dim, dtype=complex)
    if kind == "number":
        number = int(param_photon.get("n", 1))
        
        if number < 1 or number > basis.cap:
            raise ValueError("number-state n must satisfy 1 <= n <= n_max")
        
        for m, amplitude in enumerate(packet):
            occupation = [0] * basis.modes
            occupation[m] = number
            i = basis.index.get(tuple(occupation))
            if i is not None:
                v[2 * i:2 * i + 2] += amplitude * atom_vector
    elif kind == "coherent":
        alpha = complex(param_photon.get("alpha", 1.0))
        for i, occupation in enumerate(basis.states):
            coefficient = np.exp(-abs(alpha) ** 2 / 2)
            for m, n in enumerate(occupation):
                coefficient *= (alpha * packet[m]) ** n / np.sqrt(factorial(n))
            v[2 * i:2 * i + 2] = coefficient * atom_vector
    else:
        raise ValueError("photon state must be number or coherent")

    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("The initial state has no support in this basis")
    return qutip.Qobj(v / norm)
