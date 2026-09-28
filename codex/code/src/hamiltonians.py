"""Projected Rabi Hamiltonian, built from allowed Fock transitions."""

import numpy as np
import qutip
from scipy.sparse import coo_matrix, diags


class Hamiltonian:
    def __init__(self, basis, k_tab, param_atom, RWA=False):
        self.basis = basis
        self.k_tab = np.asarray(k_tab)
        self.param_atom = param_atom
        self.RWA = RWA
        self.H0 = self.free()
        self.V = self.interaction()

    def free(self):
        omega0 = self.param_atom["omega_0"]
        energies = np.zeros(self.basis.dim)
        for i, occupation in enumerate(self.basis.states):
            field_energy = np.dot(np.abs(self.k_tab), occupation)
            energies[2 * i] = field_energy
            energies[2 * i + 1] = field_energy + omega0
        return diags(energies, format="csr")

    def interaction(self):
        L = self.param_atom["L"]
        x_tls = self.param_atom["x_tls"]
        profile = self.param_atom["coupling"]
        if profile == "sqrt":
            form_factor = np.sqrt(np.abs(self.k_tab))
        elif profile == "flat":
            form_factor = np.ones(self.basis.modes)
        else:
            raise ValueError("coupling must be 'flat' or 'sqrt'")
        g_unit = 1j * form_factor * np.exp(-1j * self.k_tab * x_tls) / np.sqrt(L)

        rows, cols, values = [], [], []
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
                        values.append(g_unit[m] * np.sqrt(occupation[m]))
                    if not self.RWA or atom == 1:
                        target = list(occupation)
                        target[m] += 1
                        j = self.basis.index.get(tuple(target))
                        if j is not None:
                            rows.append(2 * j + 1 - atom)
                            cols.append(col)
                            values.append(g_unit[m].conjugate() * np.sqrt(occupation[m] + 1))

        return coo_matrix((values, (rows, cols)),
                          shape=(self.basis.dim, self.basis.dim)).tocsr()

    def build_hamiltonian(self, D=None):
        if D is None:
            D = self.param_atom["D"]
        return qutip.Qobj(self.H0 + D * self.V)
