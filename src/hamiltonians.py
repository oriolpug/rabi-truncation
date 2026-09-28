"""Projected H = H0 + D V, with u_m = i f_m exp(-i k_m x_tls) / sqrt(L)."""

import numpy as np
import qutip
from scipy.sparse import coo_matrix, diags


class Hamiltonian:
    def __init__(self, basis, k_tab, param_atom, RWA=False):
        self.basis = basis
        self.k_tab = np.asarray(k_tab)
        self.param_atom = param_atom
        self.RWA = RWA
        L, omega, x = (float(param_atom[key]) for key in ("L", "omega_0", "x_tls"))
        if not np.isfinite([L, omega, x]).all() or L <= 0 or omega < 0:
            raise ValueError("L must be positive, omega_0 nonnegative, and x_tls finite")
        self.H0 = self.free()
        self.V = self.interaction()

    def free(self):
        energies = np.zeros(self.basis.dim)
        for i, occupation in enumerate(self.basis.states):
            field_energy = np.dot(np.abs(self.k_tab), occupation)
            energies[2 * i] = field_energy
            energies[2 * i + 1] = field_energy + self.param_atom["omega_0"]
        return diags(energies, format="csr")

    def interaction(self):
        profile = self.param_atom["coupling"]
        if profile == "sqrt":
            form_factor = np.sqrt(np.abs(self.k_tab))
        elif profile == "flat":
            form_factor = np.ones(self.basis.modes)
        else:
            raise ValueError("coupling must be flat or sqrt")
        u = 1j * form_factor * np.exp(-1j * self.k_tab * self.param_atom["x_tls"])
        u /= np.sqrt(self.param_atom["L"])
        rows, cols, values, mode_ids = [], [], [], []
        for i, occupation in enumerate(self.basis.states):
            for atom in (0, 1):
                col = 2 * i + atom
                for m in range(self.basis.modes):
                    if occupation[m] and (not self.RWA or atom == 0):
                        target = list(occupation)
                        target[m] -= 1
                        j = self.basis.index[tuple(target)]
                        rows.append(2 * j + 1 - atom)
                        cols.append(col)
                        values.append(u[m] * np.sqrt(occupation[m]))
                        mode_ids.append(m)
                    if not self.RWA or atom == 1:
                        target = list(occupation)
                        target[m] += 1
                        j = self.basis.index.get(tuple(target))
                        if j is not None:
                            rows.append(2 * j + 1 - atom)
                            cols.append(col)
                            values.append(u[m].conjugate() * np.sqrt(occupation[m] + 1))
                            mode_ids.append(m)
        self.transition_rows = np.asarray(rows, dtype=int)
        self.transition_cols = np.asarray(cols, dtype=int)
        self.transition_values = np.asarray(values, dtype=complex)
        self.transition_modes = np.asarray(mode_ids, dtype=int)
        return coo_matrix((values, (rows, cols)), shape=(self.basis.dim, self.basis.dim)).tocsr()

    def build_hamiltonian(self, D=None):
        D = self.param_atom["D"] if D is None else D
        if np.iscomplexobj(D) or not np.isfinite(D):
            raise ValueError("D must be finite and real")
        return qutip.Qobj(self.H0 + float(D) * self.V)
