"""Tests for hamiltonians.py: HamiltonianFull, HamiltonianTruncated, HamiltonianAtom."""
import numpy as np
import pytest
import scipy.sparse as sp
import qutip

from utilities import Config, NumberState
from hamiltonians import (
    HamiltonianFull, HamiltonianTruncated, HamiltonianAtom, hamiltonian,
    Hamiltonian,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(truncation='full', modes=1, excitation_cap=3, **kwargs):
    return Config(modes=modes, excitation_cap=excitation_cap, truncation=truncation,
                  state=NumberState(1), **kwargs)


def built(cls, cfg):
    h = cls(cfg)
    h.build_hamiltonian()
    return h


# ---------------------------------------------------------------------------
# Dimension
# ---------------------------------------------------------------------------

class TestComputeDim:
    def test_full_single_mode(self):
        cfg = make_cfg('full', modes=1, excitation_cap=3)
        h = HamiltonianFull(cfg)
        assert h.compute_dim() == 2 * (3 + 1) ** 1  # 8

    def test_truncated_single_mode(self):
        cfg = make_cfg('truncated', modes=1, excitation_cap=3)
        h = HamiltonianTruncated(cfg)
        assert h.compute_dim() == 2 * (1 * 3 + 1)  # 8

    def test_atom_single_mode(self):
        cfg = make_cfg('truncated+atom', modes=1, excitation_cap=3)
        h = HamiltonianAtom(cfg)
        assert h.compute_dim() == 2 * (1 * 3 + 1) * (3 + 1)  # 32

    @pytest.mark.parametrize("truncation,cls", [
        ('full', HamiltonianFull),
        ('truncated', HamiltonianTruncated),
        ('truncated+atom', HamiltonianAtom),
    ])
    def test_matrix_shape_matches_dim(self, truncation, cls):
        cfg = make_cfg(truncation)
        h = cls(cfg)
        d = h.compute_dim()
        assert h.H.shape == (d, d)


# ---------------------------------------------------------------------------
# g() coupling
# ---------------------------------------------------------------------------

class TestCoupling:
    def test_g_formula(self):
        cfg = make_cfg('full', g=0.01)
        h = HamiltonianFull(cfg)
        k = 1.0
        assert np.isclose(h.g(k), 0.01 * np.sqrt(1.0))

    def test_g_zero_at_zero_k(self):
        cfg = make_cfg('full')
        h = HamiltonianFull(cfg)
        assert h.g(0) == 0.0

    def test_g_scales_with_sqrt_k(self):
        cfg = make_cfg('full', g=1.0)
        h = HamiltonianFull(cfg)
        assert np.isclose(h.g(4.0), 2.0)


# ---------------------------------------------------------------------------
# transition_possible
# ---------------------------------------------------------------------------

class TestTransitionPossible:
    def setup_method(self):
        self.cfg = make_cfg('full', modes=1, excitation_cap=3)
        self.h = HamiltonianFull(self.cfg)

    def test_diagonal_is_false(self):
        s = {'n1': 1, 'atom': 'g'}
        assert self.h.transition_possible(s, s) is False

    def test_same_atom_different_photon_is_false(self):
        ket = {'n1': 1, 'atom': 'g'}
        bra = {'n1': 0, 'atom': 'g'}
        assert self.h.transition_possible(ket, bra) is False

    def test_single_photon_emission_is_true(self):
        # Atom de-excites, photon created: atom e->g, n+1
        ket = {'n1': 1, 'atom': 'g'}
        bra = {'n1': 0, 'atom': 'e'}
        assert self.h.transition_possible(ket, bra) is True

    def test_single_photon_absorption_is_true(self):
        # Atom excites, photon absorbed: atom g->e, n-1
        ket = {'n1': 0, 'atom': 'e'}
        bra = {'n1': 1, 'atom': 'g'}
        assert self.h.transition_possible(ket, bra) is True

    def test_two_photon_change_is_false(self):
        ket = {'n1': 2, 'atom': 'e'}
        bra = {'n1': 0, 'atom': 'g'}
        assert self.h.transition_possible(ket, bra) is False

    def test_rwa_forbids_counter_rotating(self):
        cfg = make_cfg('full', modes=1, excitation_cap=3, RWA=True)
        h = HamiltonianFull(cfg)
        # Counter-rotating: atom excites AND photon created (energy non-conserving)
        ket = {'n1': 1, 'atom': 'e'}
        bra = {'n1': 0, 'atom': 'g'}
        assert h.transition_possible(ket, bra) is False

    def test_rwa_allows_resonant(self):
        cfg = make_cfg('full', modes=1, excitation_cap=3, RWA=True)
        h = HamiltonianFull(cfg)
        # Resonant: atom excites, photon absorbed (atom_change=+1, photon_diff=-1, sum=0)
        ket = {'n1': 0, 'atom': 'e'}
        bra = {'n1': 1, 'atom': 'g'}
        assert h.transition_possible(ket, bra) is True


# ---------------------------------------------------------------------------
# transition_sign
# ---------------------------------------------------------------------------

class TestTransitionSign:
    def setup_method(self):
        self.cfg = make_cfg('full', modes=1, excitation_cap=3)
        self.h = HamiltonianFull(self.cfg)

    def test_emission_sign(self):
        # Photon created (ket has more photons): diff = +1
        ket = {'n1': 1, 'atom': 'g'}
        bra = {'n1': 0, 'atom': 'e'}
        assert self.h.transition_sign(ket, bra) == 1

    def test_absorption_sign(self):
        # Photon absorbed (ket has fewer photons): diff = -1
        ket = {'n1': 0, 'atom': 'e'}
        bra = {'n1': 1, 'atom': 'g'}
        assert self.h.transition_sign(ket, bra) == -1


# ---------------------------------------------------------------------------
# Hermiticity and structure after build
# ---------------------------------------------------------------------------

class TestHermiticity:
    @pytest.mark.parametrize("truncation,cls", [
        ('full', HamiltonianFull),
        ('truncated', HamiltonianTruncated),
        ('truncated+atom', HamiltonianAtom),
    ])
    def test_hermitian(self, truncation, cls):
        cfg = make_cfg(truncation)
        h = built(cls, cfg)
        M = h.H.toarray()
        np.testing.assert_allclose(M, M.conj().T, atol=1e-12)

    @pytest.mark.parametrize("truncation,cls", [
        ('full', HamiltonianFull),
        ('truncated', HamiltonianTruncated),
        ('truncated+atom', HamiltonianAtom),
    ])
    def test_real_diagonal(self, truncation, cls):
        cfg = make_cfg(truncation)
        h = built(cls, cfg)
        diag = h.H.toarray().diagonal()
        np.testing.assert_allclose(diag.imag, 0, atol=1e-12)

    @pytest.mark.parametrize("truncation,cls", [
        ('full', HamiltonianFull),
        ('truncated', HamiltonianTruncated),
        ('truncated+atom', HamiltonianAtom),
    ])
    def test_vacuum_ground_energy_zero(self, truncation, cls):
        """Ground state with no photons has zero free energy."""
        cfg = make_cfg(truncation, atom_state='g')
        h = built(cls, cfg)
        idx = h.state_to_index({'n1': 0, 'atom': 'g'})
        assert np.isclose(h.H[idx, idx], 0.0, atol=1e-12)

    def test_zero_coupling_gives_diagonal(self):
        """With g=0 the interaction is zero and the matrix is diagonal."""
        cfg = make_cfg('full', g=0.0)
        h = built(HamiltonianFull, cfg)
        M = h.H.toarray()
        off_diag = M - np.diag(M.diagonal())
        np.testing.assert_allclose(np.abs(off_diag), 0, atol=1e-12)


# ---------------------------------------------------------------------------
# hamiltonian() factory
# ---------------------------------------------------------------------------

class TestHamiltonianFactory:
    @pytest.mark.parametrize("truncation", ['full', 'truncated', 'truncated+atom'])
    def test_returns_qobj(self, truncation):
        cfg = make_cfg(truncation)
        H = hamiltonian(cfg)
        assert isinstance(H, qutip.Qobj)

    @pytest.mark.parametrize("truncation,cls", [
        ('full', HamiltonianFull),
        ('truncated', HamiltonianTruncated),
        ('truncated+atom', HamiltonianAtom),
    ])
    def test_correct_dimension(self, truncation, cls):
        cfg = make_cfg(truncation)
        H = hamiltonian(cfg)
        d = cls(cfg).compute_dim()
        assert H.shape == (d, d)

    @pytest.mark.parametrize("truncation", ['full', 'truncated', 'truncated+atom'])
    def test_hermitian_qobj(self, truncation):
        cfg = make_cfg(truncation)
        H = hamiltonian(cfg)
        M = H.full()
        np.testing.assert_allclose(M, M.conj().T, atol=1e-12)
