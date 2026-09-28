"""Exact squared state/TLS fidelities in an implicit common full basis."""

import numpy as np
import pandas as pd
import qutip
from tqdm.notebook import tqdm


def _vector(ket, basis):
    """Validate and normalize a complete joint ket for fidelity evaluation.

    Parameters
    ----------
    ket : qutip.Qobj or numpy.ndarray
        Ket (d,1) as Qobj, or a one-dimensional array (d,).
    basis : FockBasis
        Expected joint dimension d and occupation/TLS ordering.

    Returns
    -------
    numpy.ndarray
        Normalized ket (d,), v/||v||, preserving every component. Zero or
        nonfinite norm and incompatible dimensions raise ValueError. This
        normalization does not modify the original solver state.
    """
    vector = ket.full()[:, 0] if isinstance(ket, qutip.Qobj) else np.asarray(ket)
    if vector.shape != (basis.dim,):
        raise ValueError("Ket dimension does not match the Fock basis")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("A fidelity requires nonzero finite kets")
    return vector / norm


def atom_density_matrix(vector):
    """Trace out the field using the alternating TLS component convention.

    Parameters
    ----------
    vector : array_like of complex
        Joint vector (2*B,), v[2*i+s]=C_(i,s), s=0:g and 1:e.

    Returns
    -------
    numpy.ndarray
        Complex matrix (2,2) in (g,e) order:
        rho_(s,s') = sum_i C_(i,s)*C_(i,s').conj() = (C.T @ C.conj())_(s,s').
        The trace is ||v||**2. Input is not normalized by this helper; divide
        by the squared ket norm when a unit-trace density matrix is needed.
    """
    coefficients = np.asarray(vector).reshape(-1, 2)
    return coefficients.T @ coefficients.conj()


def _mode_maps(k_a, k_b):
    """Map two physical momentum lists into one implicit union of field modes.

    Parameters
    ----------
    k_a, k_b : array_like of float
        Shapes (M_a,) and (M_b,). Each entry denotes a distinct physical mode;
        callers must ensure compatible box lengths and physical conventions.

    Returns
    -------
    map_a, map_b : tuple[list[int], list[int]]
        Local mode index -> union index maps, of lengths M_a and M_b. Union
        starts in k_a order; unseen k_b entries are appended. Matching uses
        absolute tolerance 1e-12 and no relative tolerance. Multiple possible
        matches raise ValueError rather than identifying an ambiguous mode.
    """
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
    """Represent each physical occupation by its two TLS amplitudes.

    Parameters
    ----------
    basis : FockBasis
        Source occupations, each of length M.
    vector : numpy.ndarray
        Complex joint ket (basis.dim,) in alternating (g,e) order.
    mapping : list[int]
        Length M; maps local modes to indices of the common physical union.

    Returns
    -------
    dict[tuple[tuple[int, int], ...], numpy.ndarray]
        Canonical key -> TLS pair (2,). A key is the sorted tuple of
        (union_mode_index, positive_occupation) pairs; vacuum has key ().
        Omitted modes therefore carry zero occupation. Values are array views,
        and multimode occupations are retained without an ambient full vector.
    """
    amplitudes = {}
    for i, occupation in enumerate(basis.states):
        key = tuple(sorted((mapping[m], int(n)) for m, n in enumerate(occupation) if n))
        amplitudes[key] = vector[2 * i:2 * i + 2]
    return amplitudes


def compare_states(ket_a, basis_a, k_a, ket_b, basis_b, k_b):
    """Compute squared state and TLS fidelities in a common full occupation space.

    Parameters
    ----------
    ket_a, ket_b : qutip.Qobj or numpy.ndarray
        Joint kets (d_a,1)/(d_b,1) as Qobj or arrays (d_a,)/(d_b,).
    basis_a, basis_b : FockBasis
        Their occupation sets and alternating TLS indexing; caps may differ.
    k_a, k_b : array_like of float
        Physical mode lists (M_a,)/(M_b,) matching the respective tuple entries.
        Caller guarantees identical box length and Hamiltonian conventions.

    Returns
    -------
    dict[str, float]
        ``F_state`` = |<I_a psi_a | I_b psi_b>|**2 and
        ``F_atom`` = (Tr sqrt(sqrt(rho_a)*rho_b*sqrt(rho_a)))**2, clipped to [0,1]
        against roundoff. psi_a/b are individually normalized whole kets;
        rho_a/b are their field traces. QuTiP's root fidelity is squared here.

    Notes
    -----
    I_a/b pad missing physical modes with vacuum in the union full basis.
    The overlap sums TLS dot products over every shared occupation key. There
    is no normalization on that intersection, and no full-space allocation.
    The ambient full basis defines overlap only; it is not a simulated reference.
    """
    a, b = _vector(ket_a, basis_a), _vector(ket_b, basis_b)
    map_a, map_b = _mode_maps(k_a, k_b)
    components_a = _components(basis_a, a, map_a)
    components_b = _components(basis_b, b, map_b)
    overlap = sum(np.vdot(components_a[key], components_b[key])
                  for key in components_a.keys() & components_b.keys())
    rho_a, rho_b = qutip.Qobj(atom_density_matrix(a)), qutip.Qobj(atom_density_matrix(b))
    return {"F_state": float(np.clip(abs(overlap) ** 2, 0, 1)),
            "F_atom": float(np.clip(qutip.fidelity(rho_a, rho_b) ** 2, 0, 1))}


def fidelities(candidate, experiment):
    """Compare the initial and final states of two propagated simulations.

    Parameters
    ----------
    candidate, experiment : Experiment
        Propagated simulations with the same box length and final time.
        Bases, mode subsets, photon caps and output spacings may differ.
        Uses state0 and result.final_state; stored histories are unnecessary.
        Both simulations must use the same physical mode and TLS conventions.

    Returns
    -------
    pandas.DataFrame
        Rows ``F_state`` and ``F_atom``; columns ``initial`` and ``final``.
        F_state = |<I_a psi_a | I_b psi_b>|**2 in the common full basis;
        F_atom = (Tr sqrt(sqrt(rho_a)*rho_b*sqrt(rho_a)))**2, with
        rho_a/b the field traces of the individually normalized joint kets.
        Delegates both comparisons to compare_states without changing states.
    """
    if not np.isclose(candidate.param_atom["L"], experiment.param_atom["L"],
                      atol=1e-12, rtol=0):
        raise ValueError("Experiments must use the same box length for physical-mode embedding")
    if not np.isclose(candidate.times[-1], experiment.times[-1], atol=1e-12, rtol=0):
        raise ValueError("Experiments must use the same final time")
    initial = compare_states(candidate.state0, candidate.basis, candidate.k_tab,
                             experiment.state0, experiment.basis, experiment.k_tab)
    final = compare_states(candidate.result.final_state, candidate.basis, candidate.k_tab,
                           experiment.result.final_state, experiment.basis, experiment.k_tab)
    return pd.DataFrame({"initial": initial, "final": final})


def fidelities_over_time(experiment_a, experiment_b, progress = False):
    """Compare synchronized stored trajectories using the implicit full embedding.

    Parameters
    ----------
    experiment_a, experiment_b : Experiment
        Propagated objects with store_state=True, equal output-time arrays and
        box lengths equal to absolute tolerance 1e-12. Bases/mode subsets may
        differ; physical mode identity and TLS conventions must agree.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``F_state`` and ``F_atom`` float arrays (N_t,), evaluated on every
        corresponding pair of stored kets. Both are squared fidelities.

    Raises
    ------
    ValueError
        Either history is absent, times differ, or box lengths are incompatible.
    """

    print("Computing fidelity over time ...")

    if not experiment_a.store_state or not experiment_b.store_state:
        raise ValueError("Both experiments must store states for a time series")
    if not np.array_equal(experiment_a.times, experiment_b.times):
        raise ValueError("Experiments must use the same output times")
    if not np.isclose(experiment_a.param_atom["L"], experiment_b.param_atom["L"],
                      atol=1e-12, rtol=0):
        raise ValueError("Experiments must use the same box length for physical-mode embedding")
    pairs = zip(experiment_a.result.states, experiment_b.result.states)

    values = []
    
    for a, b in tqdm(
        pairs,
        total=len(experiment_a.result.states),
        disable=not progress):
        values.append(compare_states(a, experiment_a.basis, experiment_a.k_tab,
                                     b, experiment_b.basis, experiment_b.k_tab,))
        
    print("Done.")

    return {name: np.array([value[name] for value in values])
            for name in ("F_state", "F_atom")}
