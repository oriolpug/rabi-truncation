"""
Implements fidelity calculations between states resulting from different simulations
"""

import states
import numpy as np

def common_basis(state1: states.State, state2: states.State):
    known = {states.StateFull, states.StateTruncated, states.StateAtom, states.StateTotalCap}
    if type(state1) not in known:
        raise Exception(f"Unknown state type: {type(state1)}")
    if type(state2) not in known:
        raise Exception(f"Unknown state type: {type(state2)}")
    return type(state2)

def fidelity_statevector(state1: states.State , state2: states.State) -> float:
    """
    :param state1 in a basis:
    :param state2 in another basis (can be same):
    :return: The fidelity between the two states
    """

    # Determine common (smallest) basis first (Truncated < Atom < Full)
    common = common_basis(state1, state2)

    ref = state1 if isinstance(state1, common) else state2
    fidelity = 0
    for basis_element in ref.all_states():
        fidelity += np.conj(state1[basis_element]) * state2[basis_element]

    fidelity = np.abs(fidelity) ** 2
    return fidelity