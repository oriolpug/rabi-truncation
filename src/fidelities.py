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


def physical_mode_map(config_sub, config_super, atol: float = 1e-9) -> dict:
    """Map each mode index of the smaller grid to the index in the larger grid.

    Mode selection (``Config.mode_selection``) keeps a *subset* of the wave-vector grid and
    renumbers it ``0..modes-1``, so the same label ``n{m+1}`` denotes different physical
    modes in a selected vs unselected simulation. This aligns them by wave-vector: for each
    index ``j`` of ``config_sub.frequencies`` return the index ``m`` of
    ``config_super.frequencies`` with the same value.

    Raises
    ------
    ValueError
        If a sub-grid wave-vector is not present in the super-grid (not a subset).
    """
    sub = np.asarray(config_sub.frequencies, dtype=float)
    sup = np.asarray(config_super.frequencies, dtype=float)
    mapping = {}
    for j, k in enumerate(sub):
        matches = np.argwhere(np.isclose(sup, k, atol=atol)).ravel()
        if len(matches) == 0:
            raise ValueError(f"sub-grid mode k={k} absent from the super-grid "
                             f"(grids are not subset-compatible)")
        mapping[j] = int(matches[0])
    return mapping


def _relabel(label: dict, mode_map: dict) -> dict:
    """Translate a basis label's field-mode indices via ``mode_map``.

    ``atom`` and the ``n_atom`` oscillator (not a field mode) pass through unchanged.
    """
    new = {}
    for key, val in label.items():
        if key == 'atom' or key == 'n_atom':
            new[key] = val
        else:  # field mode 'n{j+1}'
            j = int(key[1:]) - 1
            new[f'n{mode_map[j] + 1}'] = val
    return new


def embed_in_grid(state_sub: states.State, config_super) -> states.State:
    """Embed a state from a smaller (selected) mode grid into a larger grid's basis.

    Returns a state of the same class as ``state_sub`` living in ``config_super``'s basis,
    with each amplitude placed at the physically matching mode (modes absent from the
    sub-grid stay in vacuum). Embedding a normalised state is an isometry, so the result is
    still normalised.
    """
    cls = type(state_sub)
    mode_map = physical_mode_map(state_sub.config, config_super)

    # Probe instance for basis size / indexing (only needs .config), as in fidelity_statevector.
    probe = cls.__new__(cls)
    probe.config = config_super

    v = np.zeros(probe.compute_dim(), dtype=complex)
    for label in state_sub.all_states():
        v[probe.state_to_index(_relabel(label, mode_map))] = state_sub[label]
    return cls.from_vector(config_super, v)


def fidelity_mode_selection(state1: states.State, state2: states.State) -> float:
    """Fidelity between two states of the same physical simulation, one mode-selected.

    The two states share a base wave-vector grid but keep different subsets of modes
    (e.g. ``mode_selection`` off vs on). Modes are aligned by physical wave-vector: the
    smaller-grid state is embedded into the larger grid, then the standard
    ``fidelity_statevector`` is applied. Reduces to plain ``fidelity_statevector`` when the
    grids are identical. Order-agnostic.

    Both states must use the same truncation scheme; cross-truncation comparison should use
    ``fidelity_statevector`` directly.
    """
    if type(state1) is not type(state2):
        raise Exception("fidelity_mode_selection requires the same truncation scheme; "
                        "use fidelity_statevector for cross-truncation comparisons")
    sub, sup = (state1, state2) if state1.config.modes <= state2.config.modes else (state2, state1)
    embedded = embed_in_grid(sub, sup.config)
    return fidelity_statevector(embedded, sup)


def fidelity_to_reference(candidate: states.State, reference: states.State) -> float:
    """Fidelity of any candidate simulation against a single ground-truth reference.

    Handles *both* mismatches at once: a different truncation scheme AND a different (subset)
    mode grid. The candidate is first embedded onto the reference's wave-vector grid
    (``embed_in_grid``, aligning modes physically), then compared with the standard
    ``fidelity_statevector``, which bridges truncation schemes via ``common_basis``.

    Requires ``candidate.config.frequencies`` to be a subset of
    ``reference.config.frequencies`` (candidates share the reference's base grid; selected
    candidates keep a subset). The reference is typically ``full+totalcap`` with
    ``mode_selection=False`` (the full-grid ground truth).
    """
    embedded = embed_in_grid(candidate, reference.config)
    return fidelity_statevector(embedded, reference)