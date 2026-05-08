"""
Implements fidelity calculations between states resulting from different simulations
"""

import states
import numpy as np

def common_basis(state1: states.State, state2: states.State):
    """Return the basis class to iterate over when computing the inner product.

    Requirement: every label yielded by `<class>.all_states()` must index correctly
    into both state1 and state2. StateTruncated labels ({'n_m': k, 'atom': s})
    embed unambiguously into Full / Atom (with n_atom=0) / TotalCap (other modes=0),
    so it is the universal common subspace for heterogeneous comparisons.
    """
    known = {states.StateFull, states.StateTruncated, states.StateAtom, states.StateTotalCap}
    t1, t2 = type(state1), type(state2)
    if t1 not in known:
        raise Exception(f"Unknown state type: {t1}")
    if t2 not in known:
        raise Exception(f"Unknown state type: {t2}")
    if t1 is t2:
        return t1
    return states.StateTruncated

def fidelity_statevector(state1: states.State , state2: states.State) -> float:
    """
    :param state1 in a basis:
    :param state2 in another basis (can be same):
    :return: The fidelity between the two states
    """

    common = common_basis(state1, state2)

    if isinstance(state1, common) and isinstance(state2, common):
        ref = state1 if state1.compute_dim() <= state2.compute_dim() else state2
    elif isinstance(state1, common):
        ref = state1
    elif isinstance(state2, common):
        ref = state2
    else:
        # Neither state is in the common basis class: construct a minimal
        # instance just for basis iteration (all_states / state_to_index only
        # depend on .config).
        ref = common.__new__(common)
        ref.config = state1.config

    fidelity = 0
    for basis_element in ref.all_states():
        fidelity += np.conj(state1[basis_element]) * state2[basis_element]

    fidelity = np.abs(fidelity) ** 2
    return fidelity