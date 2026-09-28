"""One momentum-grid resolver for evolution and resource estimation."""

from numbers import Integral

import numpy as np


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def momentum_modes(param_atom, cutoffs=None, CTRL_M_EXPLICIT=False, M=None):
    """Sorted signed momenta; no automatic zero-mode removal."""
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
    """Packet/resonance windows, with a nearest-mode fallback per window."""
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
    base = momentum_modes(config.param_atom, config.cutoffs, config.CTRL_M_EXPLICIT, config.M)
    if config.mode_selection:
        mask = select_modes(base, config.param_photon, config.param_atom,
                            config.photon_window, config.atom_window)
        return base, base[mask]
    return base, base.copy()


def grid_summary(base, selected, L):
    """Effective cutoffs and exact representability under either control."""
    def describe(k_tab):
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
