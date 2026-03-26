"""Tests for utilities.py: Config, calculate_wave_vectors, purity, entropy."""
import numpy as np
import pytest
import qutip

from utilities import Config, NumberState, CoherentState, calculate_wave_vectors, purity, entropy


# ---------------------------------------------------------------------------
# calculate_wave_vectors
# ---------------------------------------------------------------------------

class TestCalculateWaveVectors:
    def test_output_length(self):
        ks = calculate_wave_vectors(8, 10 * np.pi)
        assert len(ks) == 8

    def test_output_is_real(self):
        ks = calculate_wave_vectors(8, 10 * np.pi)
        assert np.all(np.isreal(ks))

    def test_values_within_nyquist_band(self):
        n, L = 16, 10 * np.pi
        ks = calculate_wave_vectors(n, L)
        k_max = n * np.pi / L
        assert np.all(np.abs(ks) <= k_max + 1e-12)

    def test_evenly_spaced(self):
        ks = calculate_wave_vectors(8, 10 * np.pi)
        spacing = np.diff(np.sort(ks))
        assert np.allclose(spacing, spacing[0], atol=1e-12)

    def test_contains_zero(self):
        # For even n the zero mode is present before Config removes it
        ks = calculate_wave_vectors(8, 10 * np.pi)
        assert 0.0 in ks


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_creation(self):
        cfg = Config(modes=1)
        assert cfg.modes == 1
        assert cfg.excitation_cap == 3

    def test_single_mode_frequency_equals_k_photon(self):
        cfg = Config(modes=1)
        assert cfg.frequencies == [cfg.k_photon]

    def test_atom_coeffs_ground(self):
        cfg = Config(modes=1, atom_state='g')
        assert cfg.atom_coeffs == {'g': 1, 'e': 0}

    def test_atom_coeffs_excited(self):
        cfg = Config(modes=1, atom_state='e')
        assert cfg.atom_coeffs == {'g': 0, 'e': 1}

    def test_atom_coeffs_plus(self):
        cfg = Config(modes=1, atom_state='+')
        assert np.isclose(cfg.atom_coeffs['g'], 1 / np.sqrt(2))
        assert np.isclose(cfg.atom_coeffs['e'], 1 / np.sqrt(2))

    def test_multimode_modes_decremented_when_zero_removed(self):
        # Config decrements modes when the zero frequency is dropped
        cfg = Config(modes=8)
        assert cfg.modes == 7  # zero mode removed

    def test_multimode_ks_has_no_zero(self):
        # cfg.ks holds the zero-free frequency array
        cfg = Config(modes=8)
        assert 0.0 not in cfg.ks

    def test_state_field_stored(self):
        st = NumberState(2)
        cfg = Config(modes=1, state=st)
        assert cfg.state is st

    def test_rwa_default_false(self):
        cfg = Config(modes=1)
        assert cfg.RWA is False


# ---------------------------------------------------------------------------
# purity
# ---------------------------------------------------------------------------

class TestPurity:
    def test_pure_state_purity_one(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        assert np.isclose(purity(rho), 1.0, atol=1e-12)

    def test_maximally_mixed_purity(self):
        rho = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        assert np.isclose(purity(rho), 0.5, atol=1e-12)

    def test_accepts_qobj(self):
        rho = qutip.Qobj(np.array([[1, 0], [0, 0]], dtype=complex))
        assert np.isclose(purity(rho), 1.0, atol=1e-12)

    def test_purity_between_zero_and_one(self):
        rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
        p = purity(rho)
        assert 0.0 <= float(np.real(p)) <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_pure_state_entropy_zero(self):
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        assert np.isclose(entropy(rho), 0.0, atol=1e-10)

    def test_maximally_mixed_entropy(self):
        rho = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        assert np.isclose(entropy(rho), np.log(2), atol=1e-10)

    def test_accepts_qobj(self):
        rho = qutip.Qobj(np.array([[0.5, 0], [0, 0.5]], dtype=complex))
        assert np.isclose(entropy(rho), np.log(2), atol=1e-10)

    def test_entropy_nonnegative(self):
        rho = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
        assert entropy(rho) >= -1e-10
