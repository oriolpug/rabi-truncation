"""Exact squared state/TLS fidelities in an implicit common full basis."""

import numpy as np
import qutip


def _vector(ket, basis):
    vector = ket.full()[:, 0] if isinstance(ket, qutip.Qobj) else np.asarray(ket)
    if vector.shape != (basis.dim,):
        raise ValueError("Ket dimension does not match the Fock basis")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("A fidelity requires nonzero finite kets")
    return vector / norm


def atom_density_matrix(vector):
    coefficients = np.asarray(vector).reshape(-1, 2)
    return coefficients.T @ coefficients.conj()


def _mode_maps(k_a, k_b):
    """Align physical modes, allowing roundoff in equivalent grids."""
    union = list(np.asarray(k_a, dtype=float))
    map_a = list(range(len(union)))
    map_b = []
    for k in k_b:
        matches = np.flatnonzero(np.isclose(union, k, atol=1e-12, rtol=0))
        if len(matches) > 1:
            raise ValueError("Ambiguous physical momentum alignment")
        if len(matches):
            map_b.append(int(matches[0]))
        else:
            map_b.append(len(union))
            union.append(float(k))
    return map_a, map_b


def _components(basis, vector, mapping):
    amplitudes = {}
    for i, occupation in enumerate(basis.states):
        key = tuple(sorted((mapping[m], int(n)) for m, n in enumerate(occupation) if n))
        amplitudes[key] = vector[2 * i:2 * i + 2]
    return amplitudes


def compare_states(ket_a, basis_a, k_a, ket_b, basis_b, k_b):
    """Isometric embedding without allocating the ambient full-space vectors."""
    a, b = _vector(ket_a, basis_a), _vector(ket_b, basis_b)
    map_a, map_b = _mode_maps(k_a, k_b)
    components_a = _components(basis_a, a, map_a)
    components_b = _components(basis_b, b, map_b)
    overlap = sum(np.vdot(components_a[key], components_b[key])
                  for key in components_a.keys() & components_b.keys())
    rho_a, rho_b = qutip.Qobj(atom_density_matrix(a)), qutip.Qobj(atom_density_matrix(b))
    return {"F_state": float(np.clip(abs(overlap) ** 2, 0, 1)),
            "F_atom": float(np.clip(qutip.fidelity(rho_a, rho_b) ** 2, 0, 1))}


def fidelities_over_time(experiment_a, experiment_b):
    if not experiment_a.store_state or not experiment_b.store_state:
        raise ValueError("Both experiments must store states for a time series")
    if not np.array_equal(experiment_a.times, experiment_b.times):
        raise ValueError("Experiments must use the same output times")
    if not np.isclose(experiment_a.param_atom["L"], experiment_b.param_atom["L"],
                      atol=1e-12, rtol=0):
        raise ValueError("Experiments must use the same box length for physical-mode embedding")
    pairs = zip(experiment_a.result.states, experiment_b.result.states)
    values = [compare_states(a, experiment_a.basis, experiment_a.k_tab,
                             b, experiment_b.basis, experiment_b.k_tab) for a, b in pairs]
    return {name: np.array([value[name] for value in values])
            for name in ("F_state", "F_atom")}
