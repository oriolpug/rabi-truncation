"""
Tests for states.py — covers both the standard StateType initialiser and the
new State.from_vector classmethod, as well as the state() factory function.
"""
import numpy as np
import pytest
import qutip

from utilities import Config, NumberState, CoherentState
from states import StateFull, StateTruncated, StateAtom, State, state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(truncation='full', modes=1, excitation_cap=3, atom_state='g', state_type=None):
    st = state_type or NumberState(1)
    return Config(modes=modes, excitation_cap=excitation_cap, truncation=truncation,
                  atom_state=atom_state, state=st)


def is_normalized(v, atol=1e-12):
    return np.isclose(np.linalg.norm(v), 1.0, atol=atol)


def get_dim(truncation, modes=1, excitation_cap=3):
    cls = {'full': StateFull, 'truncated': StateTruncated, 'truncated+atom': StateAtom}[truncation]
    cfg = make_cfg(truncation, modes=modes, excitation_cap=excitation_cap)
    return cls(cfg, NumberState(1)).compute_dim()


# ---------------------------------------------------------------------------
# compute_dim
# ---------------------------------------------------------------------------

class TestComputeDim:
    def test_full_single_mode(self):
        cfg = make_cfg('full', modes=1, excitation_cap=3)
        s = StateFull(cfg, NumberState(1))
        assert s.compute_dim() == 2 * (3 + 1) ** 1  # 8

    def test_truncated_single_mode(self):
        cfg = make_cfg('truncated', modes=1, excitation_cap=3)
        s = StateTruncated(cfg, NumberState(1))
        assert s.compute_dim() == 2 * (1 * 3 + 1)  # 8

    def test_atom_single_mode(self):
        cfg = make_cfg('truncated+atom', modes=1, excitation_cap=3)
        s = StateAtom(cfg, NumberState(1))
        N, M = 3, 1
        assert s.compute_dim() == 2 * (M * N + 1) * (N + 1)  # 32

    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_vector_length_matches_dim(self, truncation, cls):
        cfg = make_cfg(truncation)
        s = cls(cfg, NumberState(1))
        assert len(s.v) == s.compute_dim()


# ---------------------------------------------------------------------------
# state_to_index
# ---------------------------------------------------------------------------

class TestStateToIndex:
    def test_full_vacuum_ground(self):
        cfg = make_cfg('full')
        s = StateFull(cfg, NumberState(0))
        assert s.state_to_index({'n1': 0, 'atom': 'g'}) == 0

    def test_full_vacuum_excited(self):
        cfg = make_cfg('full')
        s = StateFull(cfg, NumberState(0))
        assert s.state_to_index({'n1': 0, 'atom': 'e'}) == 1

    def test_full_n1_ground(self):
        cfg = make_cfg('full')
        s = StateFull(cfg, NumberState(1))
        assert s.state_to_index({'n1': 1, 'atom': 'g'}) == 2

    def test_full_n1_excited(self):
        cfg = make_cfg('full')
        s = StateFull(cfg, NumberState(1))
        assert s.state_to_index({'n1': 1, 'atom': 'e'}) == 3

    def test_full_all_indices_in_range(self):
        cfg = make_cfg('full')
        s = StateFull(cfg, NumberState(1))
        dim = s.compute_dim()
        for st in s.all_states():
            assert 0 <= s.state_to_index(st) < dim

    def test_full_all_indices_unique(self):
        cfg = make_cfg('full')
        s = StateFull(cfg, NumberState(1))
        indices = [s.state_to_index(st) for st in s.all_states()]
        assert len(indices) == len(set(indices))

    def test_truncated_vacuum_ground(self):
        cfg = make_cfg('truncated')
        s = StateTruncated(cfg, NumberState(1))
        assert s.state_to_index({'atom': 'g'}) == 0

    def test_truncated_n1_ground(self):
        cfg = make_cfg('truncated')
        s = StateTruncated(cfg, NumberState(1))
        assert s.state_to_index({'n1': 1, 'atom': 'g'}) == 2

    def test_truncated_all_indices_in_range(self):
        cfg = make_cfg('truncated')
        s = StateTruncated(cfg, NumberState(1))
        dim = s.compute_dim()
        for st in s.all_states():
            assert 0 <= s.state_to_index(st) < dim

    def test_atom_all_indices_in_range(self):
        cfg = make_cfg('truncated+atom')
        s = StateAtom(cfg, NumberState(1))
        dim = s.compute_dim()
        for st in s.all_states():
            assert 0 <= s.state_to_index(st) < dim


# ---------------------------------------------------------------------------
# Normalization of build_vector
# ---------------------------------------------------------------------------

class TestNormalization:
    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_number_state_normalized(self, truncation, cls):
        cfg = make_cfg(truncation)
        s = cls(cfg, NumberState(1))
        assert is_normalized(s.v)

    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_coherent_state_normalized(self, truncation, cls):
        cfg = make_cfg(truncation)
        s = cls(cfg, CoherentState(0.5))
        assert is_normalized(s.v)

    @pytest.mark.parametrize("atom_state", ['g', 'e', '+'])
    def test_atom_superposition_normalized(self, atom_state):
        cfg = make_cfg('full', atom_state=atom_state)
        s = StateFull(cfg, NumberState(1))
        assert is_normalized(s.v)


# ---------------------------------------------------------------------------
# calculate_ck
# ---------------------------------------------------------------------------

class TestCalculateCk:
    def test_ck_normalized(self):
        cfg = make_cfg('full')
        ck = State.calculate_ck(cfg)
        assert np.isclose(np.linalg.norm(ck), 1.0, atol=1e-14)

    def test_ck_length_equals_modes(self):
        cfg = make_cfg('full')
        ck = State.calculate_ck(cfg)
        assert len(ck) == cfg.modes


# ---------------------------------------------------------------------------
# from_vector classmethod
# ---------------------------------------------------------------------------

class TestFromVector:
    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_normalizes_input(self, truncation, cls):
        cfg = make_cfg(truncation)
        dim = cls(cfg, NumberState(1)).compute_dim()
        rng = np.random.default_rng(0)
        raw = rng.random(dim) + 1j * rng.random(dim)
        s = cls.from_vector(cfg, raw)
        assert is_normalized(s.v)

    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_preserves_direction(self, truncation, cls):
        cfg = make_cfg(truncation)
        dim = cls(cfg, NumberState(1)).compute_dim()
        raw = np.zeros(dim, dtype=complex)
        raw[0] = 3.0 + 4.0j  # single nonzero component
        s = cls.from_vector(cfg, raw)
        np.testing.assert_allclose(s.v, raw / np.linalg.norm(raw), atol=1e-14)

    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_state_type_is_none(self, truncation, cls):
        cfg = make_cfg(truncation)
        dim = cls(cfg, NumberState(1)).compute_dim()
        s = cls.from_vector(cfg, np.ones(dim, dtype=complex))
        assert s.state_type is None

    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_ck_is_none(self, truncation, cls):
        cfg = make_cfg(truncation)
        dim = cls(cfg, NumberState(1)).compute_dim()
        s = cls.from_vector(cfg, np.ones(dim, dtype=complex))
        assert s.ck is None

    def test_stores_config(self):
        cfg = make_cfg('full')
        dim = StateFull(cfg, NumberState(1)).compute_dim()
        s = StateFull.from_vector(cfg, np.ones(dim, dtype=complex))
        assert s.config is cfg

    def test_real_input_becomes_complex(self):
        cfg = make_cfg('truncated')
        dim = StateTruncated(cfg, NumberState(1)).compute_dim()
        raw = np.ones(dim)  # real dtype
        s = StateTruncated.from_vector(cfg, raw)
        assert np.iscomplexobj(s.v)

    def test_already_normalized_vector_unchanged(self):
        cfg = make_cfg('full')
        dim = StateFull(cfg, NumberState(1)).compute_dim()
        raw = np.zeros(dim, dtype=complex)
        raw[2] = 1.0
        s = StateFull.from_vector(cfg, raw)
        np.testing.assert_allclose(s.v, raw, atol=1e-14)

    def test_basis_methods_still_work(self):
        """from_vector instances should still support state_to_index and atom_density_matrix."""
        cfg = make_cfg('full')
        dim = StateFull(cfg, NumberState(1)).compute_dim()
        raw = np.zeros(dim, dtype=complex)
        raw[2] = 1.0  # |n1=1, atom=g>
        s = StateFull.from_vector(cfg, raw)
        assert s.state_to_index({'n1': 1, 'atom': 'g'}) == 2
        rho = s.atom_density_matrix()
        assert rho.shape == (2, 2)


# ---------------------------------------------------------------------------
# state() factory function
# ---------------------------------------------------------------------------

class TestStateFactory:
    @pytest.mark.parametrize("truncation", ['full', 'truncated', 'truncated+atom'])
    def test_returns_qobj(self, truncation):
        cfg = make_cfg(truncation)
        result = state(cfg)
        assert isinstance(result, qutip.Qobj)

    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_with_vector_returns_qobj(self, truncation, cls):
        cfg = make_cfg(truncation)
        dim = cls(cfg, NumberState(1)).compute_dim()
        v = np.zeros(dim, dtype=complex)
        v[0] = 1.0
        result = state(cfg, v=v)
        assert isinstance(result, qutip.Qobj)

    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_factory_with_vector_matches_from_vector(self, truncation, cls):
        cfg = make_cfg(truncation)
        dim = cls(cfg, NumberState(1)).compute_dim()
        rng = np.random.default_rng(42)
        raw = rng.random(dim) + 1j * rng.random(dim)
        np.testing.assert_allclose(
            state(cfg, v=raw).full(),
            cls.from_vector(cfg, raw).to_qobj().full(),
            atol=1e-14
        )

    @pytest.mark.parametrize("truncation", ['full', 'truncated', 'truncated+atom'])
    def test_default_path_normalized(self, truncation):
        cfg = make_cfg(truncation)
        result = state(cfg)
        norm = np.linalg.norm(result.full())
        assert np.isclose(norm, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Density matrices
# ---------------------------------------------------------------------------

class TestDensityMatrices:
    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_atom_dm_shape(self, truncation, cls):
        cfg = make_cfg(truncation)
        s = cls(cfg, NumberState(1))
        assert s.atom_density_matrix().shape == (2, 2)

    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_atom_dm_trace_one(self, truncation, cls):
        cfg = make_cfg(truncation)
        s = cls(cfg, NumberState(1))
        assert np.isclose(np.trace(s.atom_density_matrix()), 1.0, atol=1e-12)

    @pytest.mark.parametrize("truncation,cls", [
        ('full', StateFull),
        ('truncated', StateTruncated),
        ('truncated+atom', StateAtom),
    ])
    def test_atom_dm_hermitian(self, truncation, cls):
        cfg = make_cfg(truncation)
        s = cls(cfg, NumberState(1))
        rho = s.atom_density_matrix()
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-14)

    def test_atom_dm_ground_state(self):
        cfg = make_cfg('full', atom_state='g')
        s = StateFull(cfg, NumberState(1))
        rho = s.atom_density_matrix()
        assert np.isclose(rho[0, 0], 1.0, atol=1e-12)
        assert np.isclose(rho[1, 1], 0.0, atol=1e-12)

    def test_atom_dm_excited_state(self):
        cfg = make_cfg('full', atom_state='e')
        s = StateFull(cfg, NumberState(1))
        rho = s.atom_density_matrix()
        assert np.isclose(rho[0, 0], 0.0, atol=1e-12)
        assert np.isclose(rho[1, 1], 1.0, atol=1e-12)

    def test_photon_dm_shape(self):
        cfg = make_cfg('full')
        s = StateFull(cfg, NumberState(1))
        rho = s.photon_density_matrix()
        half = s.compute_dim() // 2
        assert rho.shape == (half, half)

    def test_photon_dm_trace_one(self):
        cfg = make_cfg('full', atom_state='g')
        s = StateFull(cfg, NumberState(1))
        assert np.isclose(np.trace(s.photon_density_matrix()), 1.0, atol=1e-12)

    def test_atom_dm_from_vector_trace_one(self):
        cfg = make_cfg('truncated')
        dim = StateTruncated(cfg, NumberState(1)).compute_dim()
        v = np.zeros(dim, dtype=complex)
        v[0] = 1.0  # vacuum ground
        s = StateTruncated.from_vector(cfg, v)
        assert np.isclose(np.trace(s.atom_density_matrix()), 1.0, atol=1e-12)

    def test_photon_dm_hermitian(self):
        cfg = make_cfg('full')
        s = StateFull(cfg, CoherentState(0.5))
        rho = s.photon_density_matrix()
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-14)