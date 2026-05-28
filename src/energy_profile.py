"""
Energy-profile decomposition of a time-evolved state.

Resolves the total energy E = <H> two ways at every time step:

  * E(mode): energy carried by each field mode k_m, plus the bare-atom energy
    placed at k = 0. Exact identity (each interaction term changes exactly one
    mode, so it is attributable to that mode):

        <H> = sum_m [ hbar c |k_m| <n_m> + <H_int,m> ]  +  (bare-atom energy)

  * E(excitation): site-energy partition e_i = Re(psi_i* (H psi)_i) binned by
    photon excitation number nu(i) = sum_m n_m(i) (the atom is not counted in the
    index, but atom-excited states still contribute their full energy to their bin).

Both decompositions sum to <H> to machine precision.
"""

import numpy as np
import scipy.sparse as sp

from states import StateFull, StateTruncated, StateAtom, StateTotalCap

_BASIS = {
    "truncated": StateTruncated,
    "truncated+atom": StateAtom,
    "full": StateFull,
    "full+totalcap": StateTotalCap,
}


def _to_scipy_csr(H) -> sp.csr_matrix:
    """Best-effort conversion of a qutip Qobj Hamiltonian to a scipy CSR matrix."""
    try:
        return H.data.as_scipy().tocsr()
    except AttributeError:
        return sp.csr_matrix(H.full())


class EnergyProfile:
    """Precomputes the per-mode / per-excitation structure of a Hamiltonian.

    Construct once from a built Simulation's config and Hamiltonian; then call the
    per-state-vector methods for any psi obtained during the evolution.
    """

    def __init__(self, config, H):
        self.config = config
        M = config.modes
        self.kmodes = np.array([config.frequencies[m] for m in range(M)], dtype=float)

        H_csr = _to_scipy_csr(H)
        self.H_csr = H_csr
        d = H_csr.shape[0]
        self.dim = d

        # index -> basis-state dict (same indexing the Hamiltonian/state vectors use).
        # all_states / state_to_index only need .config, so bypass State.__init__.
        basis = object.__new__(_BASIS[config.truncation])
        basis.config = config
        states = [None] * d
        for s in basis.all_states():
            states[basis.state_to_index(s)] = s
        self._states = states

        # photon number per mode and photon excitation number per basis state
        n_per_mode = np.zeros((M, d), dtype=float)
        exc = np.zeros(d, dtype=np.int64)
        for i, s in enumerate(states):
            for m in range(M):
                n = s.get(f'n{m + 1}', 0)
                if n:
                    n_per_mode[m, i] = n
                    exc[i] += n
        self.exc = exc
        self.exc_max = int(exc.max())

        # Free (diagonal) energies, read straight from H so any hbar/convention is exact.
        diag = np.real(H_csr.diagonal())
        hbar, c = config.hbar, config.c
        # Field diagonal energy per mode: hbar c |k_m| n_m
        self.field_diag = (hbar * c) * np.abs(self.kmodes)[:, None] * n_per_mode  # (M, d)
        # Whatever is left on the diagonal belongs to the atom (bare-atom + n_atom oscillator).
        self.atom_diag = diag - self.field_diag.sum(axis=0)  # (d,)

        # Interaction (off-diagonal) terms, split by the single mode each one changes.
        Hint_modes = [sp.lil_matrix((d, d), dtype=complex) for _ in range(M)]
        Hint_atom = sp.lil_matrix((d, d), dtype=complex)
        coo = H_csr.tocoo()
        for i, j, val in zip(coo.row, coo.col, coo.data):
            if i == j:
                continue
            si, sj = states[i], states[j]
            mode = self._differing_mode(si, sj)
            if mode is None:
                Hint_atom[i, j] = val
            else:
                Hint_modes[mode][i, j] = val
        self.Hint_modes = [m.tocsr() for m in Hint_modes]
        self.Hint_atom = Hint_atom.tocsr()

    def _differing_mode(self, si, sj):
        """Index of the single photon mode that differs between two basis states.

        Returns None when no photon mode differs (e.g. only the truncated+atom
        oscillator n_atom changed), routing that term to the k=0 atom bin.
        """
        mode = None
        for m in range(self.config.modes):
            if si.get(f'n{m + 1}', 0) != sj.get(f'n{m + 1}', 0):
                if mode is not None:
                    return None  # should not happen for valid single-mode transitions
                mode = m
        return mode

    # ------------------------------------------------------------------ #
    # Per-state-vector quantities
    # ------------------------------------------------------------------ #
    def energy_modes_vec(self, psi: np.ndarray):
        """Return (E_modes, E_atom): per-field-mode energy array and the k=0 atom energy."""
        probs = np.abs(psi) ** 2
        field = self.field_diag @ probs  # (M,)
        interaction = np.array([
            np.real(np.vdot(psi, Hm @ psi)) for Hm in self.Hint_modes
        ])
        E_modes = field + interaction
        E_atom = float(self.atom_diag @ probs + np.real(np.vdot(psi, self.Hint_atom @ psi)))
        return E_modes, E_atom

    def energy_excitations_vec(self, psi: np.ndarray) -> np.ndarray:
        """Return the energy binned by photon excitation number nu (length exc_max+1)."""
        Hpsi = self.H_csr @ psi
        e_i = np.real(np.conj(psi) * Hpsi)
        return np.bincount(self.exc, weights=e_i, minlength=self.exc_max + 1)

    def total_energy_vec(self, psi: np.ndarray) -> float:
        return float(np.real(np.vdot(psi, self.H_csr @ psi)))

    @property
    def mode_axis(self) -> np.ndarray:
        """Frequency grid of the field modes (E_modes is aligned with this)."""
        return self.kmodes

    @property
    def excitation_axis(self) -> np.ndarray:
        return np.arange(self.exc_max + 1)
