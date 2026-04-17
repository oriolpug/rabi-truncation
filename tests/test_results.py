"""
End-to-end tests comparing simulation output against exact analytical results
for the single-mode Jaynes-Cummings model (RWA, resonant).

Analytical formulae
-------------------
Starting from |g, n>:  P_e(t) = sin²(g·√(k·n)·t)
Starting from |e, n>:  P_e(t) = cos²(g·√(k·(n+1))·t)

where g is the coupling constant and k = k_photon = w_atom (resonance).
"""

import numpy as np
import pytest
from utilities import Config, NumberState
from simulation import Simulation

ATOL = 1e-4


def analytical_pe_from_ground(g, k, n, times):
    """P_e(t) when the atom starts in |g> with n photons."""
    return np.sin(g * np.sqrt(k * n) * times) ** 2


def analytical_pe_from_excited(g, k, n, times):
    """P_e(t) when the atom starts in |e> with n photons (couples to |g, n+1>)."""
    return np.cos(g * np.sqrt(k * (n + 1)) * times) ** 2


def analytical_pe_from_superposition(g, k, n, times):
    """P_e(t) when the atom starts in |+> or |-> with n photons.

    The |g,n> and |e,n> components live in different JC manifolds and evolve
    independently, so the result is the same for both |+> and |->:
        P_e = ½ sin²(g√(kn)·t) + ½ cos²(g√(k(n+1))·t)
    """
    return 0.5 * (np.sin(g * np.sqrt(k * n) * times) ** 2
                  + np.cos(g * np.sqrt(k * (n + 1)) * times) ** 2)


def make_jc_config(n_photons, atom_state, g=0.1, excitation_cap=5, t=30.0, dt=0.05):
    return Config(
        modes=1,
        excitation_cap=excitation_cap,
        g=g,
        RWA=True,
        truncation="full",
        state=NumberState(n_photons),
        atom_state=atom_state,
        t=t,
        dt=dt,
    )


class TestJaynesCummingsFromGround:
    """Atom starts in |g>, field in |n>."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_rabi_oscillation(self, n):
        config = make_jc_config(n_photons=n, atom_state='g')
        sim = Simulation(config)
        sim.time_evolve()

        pe_sim = np.array(sim.compute_excited_probability()).real
        pe_exact = analytical_pe_from_ground(config.g, config.k_photon, n, sim.times)

        assert pe_sim.shape == pe_exact.shape, f"Shape mismatch for |g, {n}>: {pe_sim.shape} vs {pe_exact.shape}"
        np.testing.assert_allclose(pe_sim, pe_exact, atol=ATOL,
                                   err_msg=f"Mismatch for |g, {n}> initial state")


class TestJaynesCummingsFromExcited:
    """Atom starts in |e>, field in |n>."""

    @pytest.mark.parametrize("n", [0, 1])
    def test_rabi_oscillation(self, n):
        config = make_jc_config(n_photons=n, atom_state='e', excitation_cap=max(n + 2, 5))
        sim = Simulation(config)
        sim.time_evolve()

        pe_sim = np.array(sim.compute_excited_probability()).real
        pe_exact = analytical_pe_from_excited(config.g, config.k_photon, n, sim.times)

        assert pe_sim.shape == pe_exact.shape, f"Shape mismatch for |e, {n}>: {pe_sim.shape} vs {pe_exact.shape}"
        np.testing.assert_allclose(pe_sim, pe_exact, atol=ATOL,
                                   err_msg=f"Mismatch for |e, {n}> initial state")


class TestJaynesCummingsFromSuperposition:
    """Atom starts in |+> or |->, field in |n>."""

    @pytest.mark.parametrize("atom_state", ['+', 'minus'])
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_rabi_oscillation(self, atom_state, n):
        config = make_jc_config(n_photons=n, atom_state=atom_state, excitation_cap=max(n + 2, 5))
        sim = Simulation(config)
        sim.time_evolve()

        pe_sim = np.array(sim.compute_excited_probability()).real
        pe_exact = analytical_pe_from_superposition(config.g, config.k_photon, n, sim.times)

        assert pe_sim.shape == pe_exact.shape, f"Shape mismatch for |{atom_state}, {n}>: {pe_sim.shape} vs {pe_exact.shape}"
        np.testing.assert_allclose(pe_sim, pe_exact, atol=ATOL,
                                   err_msg=f"Mismatch for |{atom_state}, {n}> initial state")


def exact_pe_single_excitation(config, times):
    """Exact P_e(t) for |e, vacuum> in the single-excitation subspace.

    Diagonalizes the (M+1)x(M+1) subspace Hamiltonian spanned by
    {|e, vac>, |g, 1_1>, ..., |g, 1_M>} and computes time evolution
    from the eigendecomposition.
    """
    M = config.modes
    freqs = config.frequencies
    H_sub = np.zeros((M + 1, M + 1), dtype=complex)
    H_sub[0, 0] = config.w_atom
    for m in range(M):
        k_m = freqs[m]
        H_sub[m + 1, m + 1] = config.c * np.abs(k_m)
        coupling = config.g * np.sqrt(np.abs(k_m))
        H_sub[0, m + 1] = coupling * np.exp(-1j * k_m * config.x_atom)
        H_sub[m + 1, 0] = coupling * np.exp(1j * k_m * config.x_atom)

    evals, evecs = np.linalg.eigh(H_sub)
    # Overlap of each eigenvector with |e, vac> (row 0)
    c0 = evecs[0, :]
    # P_e(t) = |sum_n |c0_n|^2 exp(-i E_n t)|^2
    phases = np.exp(-1j * np.outer(times, evals))  # (T, M+1)
    amplitudes = phases @ (np.abs(c0) ** 2)         # (T,)
    return np.abs(amplitudes) ** 2


class TestMultiModeSingleExcitation:
    """Multi-mode JC (RWA): atom in |e>, all modes vacuum.

    Dynamics stays in the (M+1)-dimensional single-excitation subspace,
    which is solved exactly by matrix diagonalization.
    """

    @pytest.mark.parametrize("M", [2, 3, 4])
    def test_excited_probability(self, M):
        config = Config(
            modes=M,
            length=10.0,
            excitation_cap=1,
            g=0.1,
            RWA=True,
            truncation="full",
            state=NumberState(0),
            atom_state='e',
            t=30.0,
            dt=0.05,
        )

        sim = Simulation(config)
        sim.time_evolve()

        pe_sim = np.array(sim.compute_excited_probability()).real
        pe_exact = exact_pe_single_excitation(config, sim.times)

        assert pe_sim.shape == pe_exact.shape, f"Shape mismatch for {config.modes}-mode: {pe_sim.shape} vs {pe_exact.shape}"
        np.testing.assert_allclose(pe_sim, pe_exact, atol=ATOL,
                                   err_msg=f"Mismatch for {config.modes}-mode single-excitation sector")
