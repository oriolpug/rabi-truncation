"""One momentum-grid resolver for evolution and resource estimation."""

from numbers import Integral

import numpy as np


def integer(value, name, minimum=0):
    """Validate an integer input without silently rounding or accepting booleans.

    Parameters
    ----------
    value : int or numpy.integer
        Candidate integer. Floats (including 2.0) and booleans are rejected.
    name : str
        Parameter name used in the validation error.
    minimum : int, optional
        Inclusive lower bound, default zero.

    Returns
    -------
    int
        Python integer equal to value, satisfying value >= minimum.

    Raises
    ------
    ValueError
        The input has the wrong type or lies below the bound.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def momentum_modes(param_atom, cutoffs=None, CTRL_M_EXPLICIT=False, M=None):
    """Construct the signed periodic-box momenta k_j = j * (2*pi/L).

    Parameters
    ----------
    param_atom : dict[str, object]
        Contains finite positive box length ``L`` (float).
    cutoffs : dict[str, float] or None, optional
        Radial bounds ``ir_cutoff`` and ``uv_cutoff``, with 0 <= IR <= UV.
        Required only under cutoff control; ignored under explicit-M control.
    CTRL_M_EXPLICIT : bool, optional
        True selects exactly M modes; False selects IR <= |k_j| <= UV.
    M : int or None, optional
        Positive odd mode count when explicit control is active. Its indices
        run from -(M-1)/2 to +(M-1)/2 and include zero. Otherwise ignored.

    Returns
    -------
    numpy.ndarray
        Sorted float array of shape (M_base,), including both signed momenta.
        Cutoff control includes zero exactly when IR is zero. The count follows
        from the band, rather than from the inactive M argument.

    Raises
    ------
    ValueError
        Invalid length, bounds, control flag or explicit count, or an empty band.
    """
    L = float(param_atom["L"])
    if not np.isfinite(L) or L <= 0:
        raise ValueError("L must be finite and positive")
    if not isinstance(CTRL_M_EXPLICIT, bool):
        raise ValueError("CTRL_M_EXPLICIT must be a boolean")
    spacing = 2 * np.pi / L
    if CTRL_M_EXPLICIT:
        M = integer(M, "M", minimum=1)
        if M % 2 == 0:
            raise ValueError("M must be odd for a symmetric grid containing k=0")
        half = (M - 1) // 2
        indices = np.arange(-half, half + 1)
    else:
        if cutoffs is None:
            raise ValueError("IR/UV cutoffs are required when CTRL_M_EXPLICIT=False")
        ir, uv = float(cutoffs["ir_cutoff"]), float(cutoffs["uv_cutoff"])
        if not np.isfinite(ir) or not np.isfinite(uv) or not 0 <= ir <= uv:
            raise ValueError("Cutoffs must be finite and satisfy 0 <= ir_cutoff <= uv_cutoff")
        lower = int(np.ceil(ir / spacing - 1e-12))
        upper = int(np.floor(uv / spacing + 1e-12))
        positive = np.arange(max(1, lower), upper + 1)
        zero = np.array([0], dtype=int) if ir == 0 else np.array([], dtype=int)
        indices = np.concatenate((-positive[::-1], zero, positive))
    modes = spacing * indices
    if len(modes) == 0:
        raise ValueError("The momentum band contains no modes")
    return modes


def select_modes(k_tab, param_photon, param_atom, photon_window, atom_window):
    """Select the union of packet and atomic-resonance momentum windows.

    Parameters
    ----------
    k_tab : numpy.ndarray
        Nonempty float array of shape (M_base,), containing signed momenta.
    param_photon : dict[str, float]
        Packet centre ``k_0`` and positive width ``sigma_k``.
    param_atom : dict[str, float]
        Atomic frequency ``omega_0``; resonance centres are +/-omega_0.
    photon_window : float
        Nonnegative factor w_p in |k-k_0| <= w_p*sigma_k.
    atom_window : float
        Nonnegative factor w_a for |k-omega_0| <= w_a*sigma_k or
        |k+omega_0| <= w_a*sigma_k.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape (M_base,). Each empty window contributes its
        nearest mode (first array entry in a tie), so the union is nonempty.
        A 1e-12 absolute boundary tolerance is used. Ordering is preserved;
        this optional grid restriction does not impose an incident direction.
    """
    sigma = float(param_photon["sigma_k"])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma_k must be finite and positive")
    if any(not np.isfinite(w) or w < 0 for w in (photon_window, atom_window)):
        raise ValueError("Selection windows must be finite and nonnegative")
    centres = ((param_photon["k_0"], photon_window * sigma),
               (param_atom["omega_0"], atom_window * sigma),
               (-param_atom["omega_0"], atom_window * sigma))
    mask = np.zeros(len(k_tab), dtype=bool)
    for centre, radius in centres:
        window = np.abs(k_tab - centre) <= radius + 1e-12
        if not window.any():
            window[np.argmin(np.abs(k_tab - centre))] = True
        mask |= window
    return mask


def resolve_grid(config):
    """Resolve the base grid and the actual momenta used by a configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Carries L, grid-control inputs and optional packet/resonance windows.

    Returns
    -------
    base, selected : tuple[numpy.ndarray, numpy.ndarray]
        Sorted float arrays of shapes (M_base,) and (M_selected,). Selection
        acts after base-grid construction. With selection disabled, selected
        is a separate copy of base. Both engine and estimator use this routine.
    """
    base = momentum_modes(config.param_atom, config.cutoffs, config.CTRL_M_EXPLICIT, config.M)
    if config.mode_selection:
        mask = select_modes(base, config.param_photon, config.param_atom,
                            config.photon_window, config.atom_window)
        return base, base[mask]
    return base, base.copy()


def grid_summary(base, selected, L):
    """Describe effective cutoffs and exact equivalence of the two grid controls.

    Parameters
    ----------
    base : numpy.ndarray
        Nonempty float array (M_base,) of periodic-box momenta.
    selected : numpy.ndarray
        Nonempty float array (M_selected,) retained after optional windows.
    L : float
        Positive box length; delta_k = 2*pi/L recovers integer mode indices.

    Returns
    -------
    dict[str, object]
        ``delta_k`` (float) and ``base``/``selected`` dictionaries with mode
        counts, signed extrema, effective IR=min|k|, UV=max|k|, smallest nonzero
        |k| (or None), zero presence and integer indices. ``radial_cutoffs_exact``
        tests the entire symmetric band, including holes. ``equivalent_M`` is
        an int only for a complete symmetric band containing zero; else None.
    """
    def describe(k_tab):
        """Describe one grid using its integer periodic-box labels.

        Parameters
        ----------
        k_tab : numpy.ndarray
            Nonempty float momentum array (M,); enclosing L sets delta_k.

        Returns
        -------
        dict[str, object]
            Count, extrema, zero policy, integer labels and exact representability.
            Equality with {j: j_min <= |j| <= j_max} detects radial-band completeness.
        """
        indices = np.rint(k_tab / (2 * np.pi / L)).astype(int)
        lower, upper = int(np.abs(indices).min()), int(np.abs(indices).max())
        expected = {n for n in range(-upper, upper + 1) if abs(n) >= lower}
        radial_exact = set(indices) == expected
        nonzero = np.abs(k_tab[k_tab != 0])
        return {"n_modes": len(k_tab), "k_min": float(k_tab.min()),
                "k_max": float(k_tab.max()), "ir_effective": float(np.abs(k_tab).min()),
                "uv_effective": float(np.abs(k_tab).max()),
                "smallest_nonzero_k": float(nonzero.min()) if len(nonzero) else None,
                "zero_present": bool(np.any(k_tab == 0)),
                "radial_cutoffs_exact": radial_exact,
                "equivalent_M": len(k_tab) if lower == 0 and radial_exact else None,
                "integer_indices": indices.tolist()}
    return {"delta_k": 2 * np.pi / L, "base": describe(base), "selected": describe(selected)}
