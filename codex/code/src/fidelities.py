"""Squared state, TLS, and photon fidelities between two experiments."""

import numpy as np
import qutip


def _components(experiment, vector):
    amplitudes = {}
    for i, occupation in enumerate(experiment.basis.states):
        amplitudes[occupation] = vector[2 * i:2 * i + 2]
    return amplitudes


def _atom_density_matrix(vector):
    coefficients = vector.reshape(-1, 2)
    return coefficients.T @ coefficients.conj()


def fidelities_over_time(experiment_a, experiment_b):
    """Return exact squared fidelities on a common momentum grid."""
    if not experiment_a.store_state or not experiment_b.store_state:
        raise ValueError("Both experiments must store states")
    if not np.array_equal(experiment_a.k_tab, experiment_b.k_tab):
        raise ValueError("The two experiments must use the same signed k grid")
    if not np.array_equal(experiment_a.times, experiment_b.times):
        raise ValueError("The two experiments must use the same output times")

    state, atom, photon = [], [], []
    for ket_a, ket_b in zip(experiment_a.result.states, experiment_b.result.states):
        v_a, v_b = ket_a.full()[:, 0], ket_b.full()[:, 0]
        v_a = v_a / np.linalg.norm(v_a)
        v_b = v_b / np.linalg.norm(v_b)
        a = _components(experiment_a, v_a)
        b = _components(experiment_b, v_b)
        G = np.zeros((2, 2), dtype=complex)
        for key in a.keys() & b.keys():
            G += np.outer(a[key].conj(), b[key])
        state.append(abs(np.trace(G)) ** 2)
        photon.append(np.linalg.svd(G, compute_uv=False).sum() ** 2)
        rho_a = qutip.Qobj(_atom_density_matrix(v_a))
        rho_b = qutip.Qobj(_atom_density_matrix(v_b))
        atom.append(qutip.fidelity(rho_a, rho_b) ** 2)
    return {"state": np.array(state), "atom": np.array(atom),
            "photon": np.array(photon)}
