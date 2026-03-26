"""
Diverse utility functions used in the simulations
"""

from dataclasses import dataclass, field
from math import pi
import numpy as np
from qutip import Qobj, entropy_vn

class StateType:
    name: str = ""

class NumberState(StateType):
    name: str = "number"

    def __init__(self, number: int = 1):
        self.number = number

class CoherentState(StateType):
    name: str = "coherent"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

@dataclass
class Config:
    """ Single source of truth for sim parameters"""
    # Grid
    modes: int = 64
    length: float = 10 * pi
    frequencies: list[float] = field(init=False)

    # Hamiltonian
    g: float = 0.01
    x_atom: float = length / 2
    w_atom: float = 1.0

    # State
    x_photon : float = 0.0
    k_photon : float = w_atom
    sigma_photon : float = 1.0
    state: StateType = NumberState

    # Atom
    atom_state: str = 'g'

    # Evolution
    t: float = 10.0
    dt: float = 0.1

    # Physics
    c: float = 1.0
    hbar: float = 1.0

    # Simulation type tuning
    excitation_cap : int = 3
    RWA: bool = False
    truncation: str = "full"

    def __post_init__(self):
        match self.atom_state:
            case "g":
                self.atom_coeffs = {'g':1, 'e': 0}
            case "e":
                self.atom_coeffs = {'g': 0, 'e': 1}
            case "+" | "plus":
                self.atom_coeffs = {'g': 1/np.sqrt(2), 'e': 1/np.sqrt(2)}
            case "+" | "minus":
                self.atom_coeffs = {'g': 1 / np.sqrt(2), 'e': -1 / np.sqrt(2)}
            case _:
                self.atom_coeffs = {'g': self.atom_state[0], 'e': self.atom_state[1]}

        if self.modes > 1:
            self.frequencies = calculate_wave_vectors(self.modes, self.length)
            # If k = 0 is in the vector, take it out
            zero = np.argwhere(self.frequencies == 0)
            if len(zero) > 0:
                zero = zero[0][0]
                self.ks = np.concatenate((self.frequencies[:zero], self.frequencies[zero + 1:]))
                self.modes -= 1
        else:
            # Single-mode cavity: align frequency to photon
            self.frequencies = [self.k_photon]

def calculate_wave_vectors(num_modes : int, length: float) -> np.ndarray:
    """
    Calculate wave vectors for fixed boundary conditions.

    :type num_modes: int
    :type length: float
    :param num_modes: Number of allowed modes.
    :param length: Length of the system.
    :return: Array of wave vectors `ks` for the modes.
    """
    k_max = num_modes * np.pi / length
    initial_wave_vectors = np.linspace(0, 2 * np.pi * (num_modes - 1) / length, num_modes)
    wave_vectors = np.zeros(num_modes)

    for i, k in enumerate(initial_wave_vectors):
        wave_vectors[i % num_modes] = k if k <= k_max else k - 2 * k_max

    wave_vectors = np.fft.fftshift(wave_vectors)
    if wave_vectors[0] > 0:
        wave_vectors = np.roll(wave_vectors, -1)

    return wave_vectors

def purity(rho):
    if not isinstance(rho, Qobj): # " Input needs to be a qutip density matrix (qutip.Qobj)"
        rho = Qobj(rho)
    return (rho ** 2).tr()

def entropy(rho):
    if not isinstance(rho, Qobj): # " Input needs to be a qutip density matrix (qutip.Qobj)"
        rho = Qobj(rho)
    return entropy_vn(rho)
