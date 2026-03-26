"""Tests for fidelities.py: common_basis and fidelity_statevector."""
import numpy as np
import pytest

from utilities import Config, NumberState
from states import StateFull, StateTruncated, StateAtom
from fidelities import common_basis, fidelity_statevector


def make_cfg(truncation='full', modes=1, excitation_cap=3):
    return Config(modes=modes, excitation_cap=excitation_cap, truncation=truncation,
                  state=NumberState(1))


# ---------------------------------------------------------------------------
# common_basis
# ---------------------------------------------------------------------------

class TestCommonBasis:
    @pytest.mark.parametrize("t1,cls1,t2,cls2,expected", [
        ('full',         StateFull,      'full',         StateFull,      StateFull),
        ('full',         StateFull,      'truncated',    StateTruncated, StateTruncated),
        ('full',         StateFull,      'truncated+atom', StateAtom,    StateAtom),
        ('truncated',    StateTruncated, 'full',         StateFull,      StateFull),
        ('truncated',    StateTruncated, 'truncated',    StateTruncated, StateTruncated),
        ('truncated',    StateTruncated, 'truncated+atom', StateAtom,    StateAtom),
        ('truncated+atom', StateAtom,   'full',         StateFull,      StateFull),
        ('truncated+atom', StateAtom,   'truncated',    StateTruncated, StateTruncated),
        ('truncated+atom', StateAtom,   'truncated+atom', StateAtom,    StateAtom),
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
