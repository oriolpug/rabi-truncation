"""Tests for utilities.py: Config, calculate_wave_vectors, purity, entropy."""
import numpy as np
import pytest
import qutip

from utilities import (Config, NumberState, CoherentState, calculate_wave_vectors,
                       select_modes, purity, entropy)


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

    def test_multimode_frequencies_zero_free_and_consistent(self):
        # Baseline fix: the simulation grid (frequencies) is the zero-free grid,
        # and its length matches the (decremented) mode count.
        cfg = Config(modes=8)
        assert 0.0 not in list(cfg.frequencies)
        assert len(cfg.frequencies) == cfg.modes
        np.testing.assert_array_equal(np.asarray(cfg.frequencies), np.asarray(cfg.ks))

    def test_state_field_stored(self):
        st = NumberState(2)
        cfg = Config(modes=1, state=st)
        assert cfg.state is st

    def test_rwa_default_false(self):
        cfg = Config(modes=1)
        assert cfg.RWA is False


# ---------------------------------------------------------------------------
# select_modes + Config(mode_selection=True)
# ---------------------------------------------------------------------------

class TestSelectModes:
    def test_mask_length_and_type(self):
        freqs = calculate_wave_vectors(16, 20.0)
        mask = select_modes(freqs, k_photon=1.0, sigma=1.0, w_atom=1.0,
                            photon_factor=1.5, atom_factor=0.25)
        assert mask.dtype == bool
        assert len(mask) == len(freqs)
        assert mask.any()

    def test_selected_obey_predicate_or_are_nearest(self):
        freqs = np.asarray(calculate_wave_vectors(32, 20.0))
        k0, sigma, w = 0.8, 1.0, 1.2
        pf, af = 1.5, 0.25
        mask = select_modes(freqs, k0, sigma, w, pf, af)
        centres = [(k0, pf * sigma), (w, af * sigma), (-w, af * sigma)]
        nearest = {np.argmin(np.abs(freqs - c)) for c, _ in centres}
        for i, keep in enumerate(mask):
            if not keep:
                continue
            in_window = any(abs(freqs[i] - c) <= h for c, h in centres)
            assert in_window or i in nearest

    def test_each_centre_represented(self):
        # Even with a tiny sigma (all windows narrower than the spacing), the nearest
        # mode to each of the three centres must be present.
        freqs = np.asarray(calculate_wave_vectors(32, 20.0))
        k0, sigma, w = 0.5, 1e-6, 1.0
        mask = select_modes(freqs, k0, sigma, w, 1.5, 0.25)
        for centre in (k0, w, -w):
            assert mask[np.argmin(np.abs(freqs - centre))]

    def test_config_mode_selection_subset_and_consistent(self):
        full = Config(modes=64, length=20)
        sel = Config(modes=64, length=20, mode_selection=True)
        # consistency invariant
        assert len(sel.frequencies) == sel.modes
        # strictly fewer modes, and a subset of the full zero-free grid
        assert sel.modes < full.modes
        assert set(np.round(sel.frequencies, 10)).issubset(set(np.round(full.frequencies, 10)))
        assert 0.0 not in list(sel.frequencies)

    def test_config_mode_selection_off_unchanged(self):
        # Regression guard: default path still drops exactly the zero mode.
        cfg = Config(modes=8)
        assert cfg.modes == 7
        assert cfg.mode_selection is False


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
