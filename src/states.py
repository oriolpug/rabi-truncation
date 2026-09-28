"""Fock bases and Gaussian preparations; TLS is the last binary index."""

from itertools import combinations_with_replacement, product
from math import factorial

import numpy as np
import qutip

from .grid import integer


class FockBasis:
    """Occupation tuples and joint field/TLS indexing for one finite basis.

    Attributes
    ----------
    states : list[tuple[int, ...]]
        B retained field occupations n=(n_0,...,n_{M-1}); each tuple has M entries.
    index : dict[tuple[int, ...], int]
        Inverse lookup n -> i. Joint ket component q=2*i+s uses s=0 (g), 1 (e).
    photon_numbers : numpy.ndarray
        Integer array (B,) with sum_m n_m for each occupation, excluding the TLS.
    modes, cap : int
        Retained mode count M and photon cutoff N.
    truncation : str
        Basis constraint used to enumerate states.
    dim : int
        Joint ket dimension d=2*B, including the TLS binary factor.
    """
    def __init__(self, modes, cap, truncation):
        """Enumerate the retained field occupations and their TLS indexing.

        Parameters
        ----------
        modes : int
            Positive number M of retained physical field modes.
        cap : int
            Nonnegative photon cutoff N (``n_max``).
        truncation : {'truncated', 'full+totalcap', 'full'}
            Retain vacuum plus n*e_m; all tuples with sum(n_m)<=N; or all tuples
            with each n_m<=N, respectively. Their field counts B are 1+M*N,
            binomial(M+N,N), and (N+1)**M. e_m denotes the unit occupation vector.

        Returns
        -------
        None
            Sets states, index, photon_numbers and dim on this instance. Vacuum is
            included for every scheme. Enumeration order is scheme-dependent; use
            index rather than assuming a tuple occupies the same position elsewhere.
        """
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
    """Prepare and normalize a projected Gaussian field state times a TLS state.

    Parameters
    ----------
    basis : FockBasis
        Retained occupation set, with d=2*B joint ket components.
    k_tab : array_like of float
        Shape (M,), one signed momentum per occupation entry in basis order.
    param_photon : dict[str, object]
        Finite ``k_0``, ``x_0``, positive ``sigma_k``; ``state`` is 'number'
        (default) or 'coherent'. Number preparation uses integer ``n`` (default
        1, 0<=n<=N); coherent preparation uses finite complex ``alpha`` (default 1).
    param_atom : dict[str, object]
        Optional ``initial_state``: 'g' (default), 'e', '+', or '-'; the latter
        two are (|g> +/- |e>)/sqrt(2).

    Returns
    -------
    qutip.Qobj
        Normalized ket of shape (d,1), with components q=2*i+s (s=0:g, 1:e).
        QuTiP stores flat dimensions; the occupation/TLS split is given by basis.

    Notes
    -----
    Packet amplitudes are c_m proportional to
    exp(-(k_m-k_0)**2/(4*sigma_k**2)) * exp(-i*k_m*x_0), with sum|c_m|**2=1.
    Every supplied signed mode participates. Subtracting the largest Gaussian
    exponent prevents common underflow without changing normalized amplitudes.
    For n>0, the field state is sum_m c_m |n*e_m>; for n=0 it is vacuum once.
    This n>1 convention differs from (sum_m c_m a_m^dagger)**n |0>/sqrt(n!).
    For coherent input, each retained tuple receives
    prod_m (alpha*c_m)**n_m/sqrt(n_m!), followed by whole-ket normalization.
    The omitted common exp(-|alpha|**2/2) cancels in that projection normalization.
    """
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
