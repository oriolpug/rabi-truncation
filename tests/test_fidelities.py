"""Tests for fidelities.py: common_basis and fidelity_statevector."""
import numpy as np
import pytest

from utilities import Config, NumberState, CoherentState
from states import StateFull, StateTruncated, StateAtom, StateTotalCap
from fidelities import (common_basis, fidelity_statevector,
                        physical_mode_map, embed_in_grid, fidelity_mode_selection)


def make_cfg(truncation='full', modes=1, excitation_cap=3):
    return Config(modes=modes, excitation_cap=excitation_cap, truncation=truncation,
                  state=NumberState(1))


# ---------------------------------------------------------------------------
# common_basis
# ---------------------------------------------------------------------------

class TestCommonBasis:
    @pytest.mark.parametrize("t1,cls1,t2,cls2,expected", [
        # Same type → that type
        ('full',         StateFull,      'full',         StateFull,      StateFull),
        ('truncated',    StateTruncated, 'truncated',    StateTruncated, StateTruncated),
        ('truncated+atom', StateAtom,   'truncated+atom', StateAtom,    StateAtom),
        # Heterogeneous → StateTruncated (universal common subspace)
        ('full',         StateFull,      'truncated',    StateTruncated, StateTruncated),
        ('full',         StateFull,      'truncated+atom', StateAtom,    StateTruncated),
        ('truncated',    StateTruncated, 'full',         StateFull,      StateTruncated),
        ('truncated',    StateTruncated, 'truncated+atom', StateAtom,    StateTruncated),
        ('truncated+atom', StateAtom,   'full',         StateFull,      StateTruncated),
        ('truncated+atom', StateAtom,   'truncated',    StateTruncated, StateTruncated),
    ])
    def test_all_pairs(self, t1, cls1, t2, cls2, expected):
        s1 = cls1(make_cfg(t1), NumberState(1))
        s2 = cls2(make_cfg(t2), NumberState(1))
        assert common_basis(s1, s2) is expected

    def test_unknown_type_raises(self):
        cfg = make_cfg('full')
        s1 = StateFull(cfg, NumberState(1))

        class Bogus:
            pass

        with pytest.raises(Exception):
            common_basis(Bogus(), s1)


# ---------------------------------------------------------------------------
# fidelity_statevector
# ---------------------------------------------------------------------------

class TestFidelityStatevector:
    def test_same_state_is_one(self):
        cfg = make_cfg('truncated')
        s = StateTruncated(cfg, NumberState(1))
        assert np.isclose(fidelity_statevector(s, s), 1.0, atol=1e-12)

    def test_orthogonal_states_is_zero(self):
        cfg = make_cfg('full')
        # |n=1, atom=g> vs |n=1, atom=e> are orthogonal
        cfg_g = Config(modes=1, excitation_cap=3, truncation='full',
                       atom_state='g', state=NumberState(1))
        cfg_e = Config(modes=1, excitation_cap=3, truncation='full',
                       atom_state='e', state=NumberState(1))
        sg = StateFull(cfg_g, NumberState(1))
        se = StateFull(cfg_e, NumberState(1))
        assert np.isclose(fidelity_statevector(sg, se), 0.0, atol=1e-12)

    def test_fidelity_between_zero_and_one(self):
        cfg = make_cfg('truncated')
        s1 = StateTruncated(cfg, NumberState(1))
        s2 = StateTruncated(cfg, NumberState(2))
        f = fidelity_statevector(s1, s2)
        assert 0.0 <= f <= 1.0 + 1e-12

    def test_fidelity_is_symmetric(self):
        cfg = make_cfg('truncated')
        s1 = StateTruncated(cfg, NumberState(1))
        s2 = StateTruncated(cfg, NumberState(2))
        assert np.isclose(fidelity_statevector(s1, s2), fidelity_statevector(s2, s1), atol=1e-12)

    def test_same_state_different_basis_objects(self):
        """Two independently constructed identical states should have fidelity 1."""
        cfg1 = make_cfg('truncated')
        cfg2 = make_cfg('truncated')
        s1 = StateTruncated(cfg1, NumberState(1))
        s2 = StateTruncated(cfg2, NumberState(1))
        assert np.isclose(fidelity_statevector(s1, s2), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# mode-selection comparison (wave-vector aligned)
# ---------------------------------------------------------------------------

def _unit_state(cls, cfg, pairs):
    """Build a State in cfg's basis from a list of (label_dict, amplitude) pairs."""
    probe = cls.__new__(cls)
    probe.config = cfg
    v = np.zeros(probe.compute_dim(), dtype=complex)
    for label, amp in pairs:
        v[probe.state_to_index(label)] = amp
    return cls.from_vector(cfg, v)


def _exc(mode_index, n=1, atom='g'):
    """Truncated-basis single-mode excitation label."""
    return {f'n{mode_index + 1}': n, 'atom': atom}


def _full_and_selected(truncation='truncated', modes=16, N=2):
    full = Config(modes=modes, length=20, truncation=truncation, excitation_cap=N,
                  state=NumberState(1), mode_selection=False)
    sel = Config(modes=modes, length=20, truncation=truncation, excitation_cap=N,
                 state=NumberState(1), mode_selection=True)
    return full, sel


class TestPhysicalModeMap:
    def test_identity_for_equal_grids(self):
        cfg = Config(modes=8, length=20)
        m = physical_mode_map(cfg, cfg)
        assert m == {j: j for j in range(cfg.modes)}

    def test_selected_is_subset_of_full(self):
        full, sel = _full_and_selected()
        m = physical_mode_map(sel, full)
        assert len(m) == sel.modes
        for j, mm in m.items():
            assert np.isclose(sel.frequencies[j], full.frequencies[mm])

    def test_raises_when_not_subset(self):
        a = Config(modes=8, length=20)   # spacing 2*pi/20
        b = Config(modes=8, length=10)   # different spacing -> different wave-vectors
        with pytest.raises(ValueError):
            physical_mode_map(a, b)


class TestEmbedInGrid:
    def test_preserves_norm_and_places_amplitude(self):
        full, sel = _full_and_selected()
        m = physical_mode_map(sel, full)[0]
        s_sel = _unit_state(StateTruncated, sel, [(_exc(0), 1.0)])
        embedded = embed_in_grid(s_sel, full)
        assert np.isclose(np.linalg.norm(embedded.v), 1.0)
        # amplitude landed at the physically matching full-grid mode
        assert np.isclose(abs(embedded[_exc(m)]), 1.0)


class TestFidelityModeSelection:
    def test_supported_only_on_selected_modes_is_one(self):
        full, sel = _full_and_selected()
        m = physical_mode_map(sel, full)[0]
        s_full = _unit_state(StateTruncated, full, [(_exc(m), 1.0)])
        s_sel = _unit_state(StateTruncated, sel, [(_exc(0), 1.0)])
        assert np.isclose(fidelity_mode_selection(s_full, s_sel), 1.0, atol=1e-12)

    def test_retained_probability(self):
        full, sel = _full_and_selected()
        mode_map = physical_mode_map(sel, full)
        sel_full_indices = set(mode_map.values())
        m_non = next(i for i in range(full.modes) if i not in sel_full_indices)
        m_sel = mode_map[0]
        # full state splits equally between a selected and a non-selected mode
        s_full = _unit_state(StateTruncated, full, [(_exc(m_sel), 1.0), (_exc(m_non), 1.0)])
        s_sel = _unit_state(StateTruncated, sel, [(_exc(0), 1.0)])
        assert np.isclose(fidelity_mode_selection(s_full, s_sel), 0.5, atol=1e-12)

    def test_symmetric(self):
        full, sel = _full_and_selected()
        m_sel = physical_mode_map(sel, full)[0]
        s_full = _unit_state(StateTruncated, full, [(_exc(m_sel), 1.0)])
        s_sel = _unit_state(StateTruncated, sel, [(_exc(0), 1.0)])
        assert np.isclose(fidelity_mode_selection(s_full, s_sel),
                          fidelity_mode_selection(s_sel, s_full), atol=1e-12)

    def test_raises_on_mismatched_truncation(self):
        full_f = Config(modes=4, length=20, truncation='full', excitation_cap=2)
        full_t = Config(modes=4, length=20, truncation='truncated', excitation_cap=2)
        sf = StateFull(full_f, NumberState(1))
        st = StateTruncated(full_t, NumberState(1))
        with pytest.raises(Exception):
            fidelity_mode_selection(sf, st)


class TestFidelityModeSelectionEndToEnd:
    """Through real time evolution (full+totalcap)."""

    def _states(self, mode_selection, photon_window=1.5, atom_window=0.25):
        from simulation import Simulation
        cfg = Config(modes=8, length=20, truncation='full+totalcap', excitation_cap=2,
                     g=0.1, atom_state='e', state=CoherentState(1.0), t=4.0, dt=0.2,
                     mode_selection=mode_selection,
                     photon_window=photon_window, atom_window=atom_window)
        sim = Simulation(cfg)
        sim.time_evolve()
        return [StateTotalCap.from_vector(cfg, s.full()[:, 0]) for s in sim.result.states]

    def test_wide_windows_keep_all_modes_fidelity_one(self):
        # Windows wide enough to keep every mode -> selected sim == full sim -> F == 1 for all t.
        ref = self._states(mode_selection=False)
        sel = self._states(mode_selection=True, photon_window=100.0, atom_window=100.0)
        assert len(sel[0].config.frequencies) == len(ref[0].config.frequencies)
        fids = [fidelity_mode_selection(a, b) for a, b in zip(sel, ref)]
        np.testing.assert_allclose(fids, 1.0, atol=1e-9)

    def test_narrow_windows_in_bounds(self):
        ref = self._states(mode_selection=False)
        sel = self._states(mode_selection=True, photon_window=0.3, atom_window=0.1)
        # selection actually drops modes here
        assert len(sel[0].config.frequencies) < len(ref[0].config.frequencies)
        fids = np.array([fidelity_mode_selection(a, b) for a, b in zip(sel, ref)])
        assert np.all(fids >= -1e-9) and np.all(fids <= 1.0 + 1e-9)
