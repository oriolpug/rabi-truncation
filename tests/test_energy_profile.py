"""
Tests for the energy-profile decomposition (src/energy_profile.py).

Correctness invariant: both the per-mode and per-excitation decompositions must
sum to the total energy <H> at every time step.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from utilities import Config, NumberState, CoherentState
from simulation import Simulation
from energy_profile import EnergyProfile, _to_scipy_csr


def make_sim(truncation, modes=3, N=2, g=0.08, atom="e", state=None, t=6.0, dt=0.2, RWA=False):
    config = Config(
        modes=modes,
        length=10.0,
        excitation_cap=N,
        g=g,
        RWA=RWA,
        truncation=truncation,
        state=state if state is not None else NumberState(1),
        atom_state=atom,
        t=t,
        dt=dt,
    )
    sim = Simulation(config)
    sim.time_evolve()
    return sim


BASES = ["full", "truncated", "truncated+atom", "full+totalcap"]


def _total_energy_series(sim):
    """Reference <H>(t) computed directly as Re<psi|H|psi>, independent of the decompositions."""
    ep = EnergyProfile(sim.config, sim.H)
    return np.array([ep.total_energy_vec(state.full()[:, 0]) for state in sim.result.states])


class TestModeDecompositionSumsToH:
    @pytest.mark.parametrize("truncation", BASES)
    def test_modes_sum(self, truncation):
        sim = make_sim(truncation)
        _, E_modes, _, E_atom = sim.compute_energy_profile_modes()
        total = _total_energy_series(sim)
        recomposed = E_modes.sum(axis=1) + E_atom
        np.testing.assert_allclose(recomposed, total, atol=1e-9, rtol=1e-9)


class TestExcitationDecompositionSumsToH:
    @pytest.mark.parametrize("truncation", BASES)
    def test_excitations_sum(self, truncation):
        sim = make_sim(truncation)
        _, E_exc = sim.compute_energy_profile_excitations()
        total = _total_energy_series(sim)
        np.testing.assert_allclose(E_exc.sum(axis=1), total, atol=1e-9, rtol=1e-9)


class TestDecompositionsAgree:
    """Two independent decompositions of the same <H> must match each other."""

    @pytest.mark.parametrize("truncation", BASES)
    def test_mode_and_excitation_sums_match(self, truncation):
        sim = make_sim(truncation)
        _, E_modes, _, E_atom = sim.compute_energy_profile_modes()
        _, E_exc = sim.compute_energy_profile_excitations()
        np.testing.assert_allclose(E_modes.sum(axis=1) + E_atom, E_exc.sum(axis=1),
                                   atol=1e-9, rtol=1e-9)


class TestOperatorIdentity:
    @pytest.mark.parametrize("truncation", BASES)
    def test_components_reconstruct_H(self, truncation):
        """sum_m H_m + H_atom == H (field diagonals + interaction split + atom)."""
        sim = make_sim(truncation)
        ep = EnergyProfile(sim.config, sim.H)
        d = ep.dim

        recon = sp.lil_matrix((d, d), dtype=complex)
        recon.setdiag(ep.field_diag.sum(axis=0) + ep.atom_diag)
        recon = recon.tocsr()
        for Hm in ep.Hint_modes:
            recon = recon + Hm
        recon = recon + ep.Hint_atom

        H_csr = _to_scipy_csr(sim.H)
        diff = (recon - H_csr).tocoo()
        max_err = np.abs(diff.data).max() if diff.nnz else 0.0
        assert max_err < 1e-9, f"{truncation}: ||sum H_m + H_atom - H|| = {max_err}"


class TestShapes:
    def test_profile_shapes(self):
        sim = make_sim("full", modes=3, N=2)
        kmodes, E_modes, atom_x, E_atom = sim.compute_energy_profile_modes()
        exc_axis, E_exc = sim.compute_energy_profile_excitations()
        T = len(sim.times)

        assert kmodes.shape == (sim.config.modes,)
        assert E_modes.shape == (T, sim.config.modes)
        assert E_atom.shape == (T,)
        assert atom_x == 0.0
        assert E_exc.shape == (T, len(exc_axis))

    def test_single_time_accessor(self):
        sim = make_sim("full", modes=2, N=2)
        _, E_modes, _, E_atom = sim.compute_energy_profile_modes(t=-1)
        _, E_exc = sim.compute_energy_profile_excitations(t=-1)
        assert E_modes.shape == (sim.config.modes,)
        assert np.isscalar(E_atom) or np.ndim(E_atom) == 0
        assert E_exc.ndim == 1
