"""Fock bases and Gaussian preparations; TLS is the last binary index."""

from itertools import combinations_with_replacement, product
from math import factorial

import numpy as np
import qutip

from .grid import integer


class FockBasis:
    def __init__(self, modes, cap, truncation):
        self.modes = modes = integer(modes, "modes", minimum=1)
        self.cap = cap = integer(cap, "n_max")
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
                for occupied in combinations_with_replacement(range(modes), total):
                    field_states.append(tuple(np.bincount(occupied, minlength=modes)))
        elif truncation == "full":
            field_states = list(product(range(cap + 1), repeat=modes))
        else:
            raise ValueError(f"Unknown truncation: {truncation}")
        self.states = field_states
        self.index = {occupation: i for i, occupation in enumerate(field_states)}
        self.photon_numbers = np.array([sum(n) for n in field_states])
        self.dim = 2 * len(field_states)


def initial_state(basis, k_tab, param_photon, param_atom):
    """Project a preparation onto the basis and normalize it once."""
    k0, sigma = float(param_photon["k_0"]), float(param_photon["sigma_k"])
    x0 = float(param_photon["x_0"])
    if not np.isfinite([k0, sigma, x0]).all() or sigma <= 0:
        raise ValueError("k_0 and x_0 must be finite; sigma_k must be finite and positive")
    k_tab = np.asarray(k_tab, dtype=float)
    exponent = -(k_tab - k0) ** 2 / (4 * sigma ** 2)
    packet = np.exp(exponent - exponent.max()) * np.exp(-1j * k_tab * x0)
    packet /= np.linalg.norm(packet)
    atom = param_atom.get("initial_state", "g")
    atom_coeffs = {"g": (1, 0), "e": (0, 1),
                   "+": (1 / np.sqrt(2), 1 / np.sqrt(2)),
                   "-": (1 / np.sqrt(2), -1 / np.sqrt(2))}
    if atom not in atom_coeffs:
        raise ValueError("initial_state must be g, e, +, or -")
    atom_vector = np.asarray(atom_coeffs[atom], dtype=complex)
    vector = np.zeros(basis.dim, dtype=complex)
    kind = param_photon.get("state", "number")
    if kind == "number":
        number = integer(param_photon.get("n", 1), "n")
        if number > basis.cap:
            raise ValueError("n must satisfy 0 <= n <= n_max")
        if number == 0:
            i = basis.index[(0,) * basis.modes]
            vector[2 * i:2 * i + 2] = atom_vector
        else:
            for m, amplitude in enumerate(packet):
                occupation = [0] * basis.modes
                occupation[m] = number
                i = basis.index[tuple(occupation)]
                vector[2 * i:2 * i + 2] = amplitude * atom_vector
    elif kind == "coherent":
        alpha = complex(param_photon.get("alpha", 1.0))
        if not np.isfinite(alpha):
            raise ValueError("alpha must be finite")
        for i, occupation in enumerate(basis.states):
            coefficient = 1.0 + 0j
            for m, n in enumerate(occupation):
                coefficient *= (alpha * packet[m]) ** n / np.sqrt(float(factorial(n)))
            vector[2 * i:2 * i + 2] = coefficient * atom_vector
    else:
        raise ValueError("state must be number or coherent")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("The projected initial state cannot be normalized")
    return qutip.Qobj(vector / norm)
