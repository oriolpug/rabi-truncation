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
    """
    Single source of truth for simulation parameters.

    Parameters
    ----------
    modes : int
        Number of photon field modes in the wave-vector grid. If k=0 falls on
        the grid it is removed and ``modes`` is decremented by 1. Default: 64.
    length : float
        Physical length of the cavity (sets the wave-vector spacing). Default: 10π.
    g : float
        Atom-field coupling constant. Default: 0.01.
    x_atom : float
        Position of the atom along the cavity. Default: length/2 (centre).
    w_atom : float
        Atom transition frequency (also used as the reference frequency for
        ``k_photon`` and for the ``AtomBasis`` oscillator mode). Default: 1.0.
    x_photon : float
        Centre position of the initial photon wave-packet. Default: 0.0.
    k_photon : float
        Central wave-vector of the initial photon state. Defaults to ``w_atom``
        (resonance).
    sigma_photon : float
        Width (standard deviation) of the photon wave-packet in k-space.
        Default: 1.0.
    state : StateType
        Class (not instance) specifying the initial photon state type.
        Use ``NumberState`` for a Fock state or ``CoherentState`` for a
        coherent state. Default: ``NumberState``.
    atom_state : str
        Initial atom state. Accepted values: ``'g'`` (ground), ``'e'``
        (excited), ``'+'`` / ``'plus'`` (|+⟩), ``'minus'`` (|−⟩), or a
        two-element sequence ``[cg, ce]`` for an arbitrary superposition.
        Default: ``'g'``.
    t : float
        Total evolution time. Default: 10.0.
    dt : float
        Time step for the solver output. Default: 0.1.
    c : float
        Speed of light (set to 1 for natural units). Default: 1.0.
    hbar : float
        Reduced Planck constant (set to 1 for natural units). Default: 1.0.
    excitation_cap : int
        Maximum number of photons allowed per mode (Fock-space truncation
        cap). Default: 3.
    RWA : bool
        If ``True``, apply the rotating-wave approximation (keep only
        resonant terms in the interaction). Default: ``False``.
    truncation : str
        Hilbert-space truncation scheme. One of:

        * ``"full"``            — all Fock states up to ``excitation_cap`` per
          mode; dim = 2(N+1)^M.
        * ``"truncated"``       — vacuum + single-mode excitations only;
          dim = 2(MN+1).
        * ``"truncated+atom"``  — truncated basis extended with an atom
          oscillator (``n_atom``) mode; dim = 2(MN+1)(N+1).
        * ``"full+totalcap"``   — all Fock states where the **total** photon
          number across all modes is ≤ ``excitation_cap``; dim = 2·C(N+M, M).
          Scales polynomially in M (unlike ``"full"``), making large mode
          counts tractable.

        Default: ``"full"``.

    Attributes set by ``__post_init__``
    ------------------------------------
    frequencies : np.ndarray
        Full wave-vector grid (including zero if present). In single-mode
        operation this is ``[k_photon]``.
    ks : np.ndarray
        Zero-free wave-vector grid used by the Hamiltonian (only set when
        ``modes > 1`` and k=0 is present).
    atom_coeffs : dict
        Mapping ``{'g': cg, 'e': ce}`` derived from ``atom_state``.
    """
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
