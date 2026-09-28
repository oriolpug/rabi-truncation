"""Projected H = H0 + D V, with u_m = i f_m exp(-i k_m x_tls) / sqrt(L)."""

import numpy as np
import qutip
from scipy.sparse import coo_matrix, diags


class Hamiltonian:
    """Sparse projected Hamiltonian H(D)=H0+D*V in natural units hbar=c=1.

    Attributes
    ----------
    basis : FockBasis
        Occupation tuples and alternating ground/excited joint indices.
    k_tab : numpy.ndarray
        Signed momenta (M,); field frequencies are |k_m|.
    param_atom : dict[str, object]
        Physical parameters retained by reference, including D and coupling.
    RWA : bool
        Whether only excitation-conserving interaction terms are retained.
    H0, V : scipy.sparse.csr_matrix
        Operators of shape (d,d), d=basis.dim. V excludes the physical scale D.
    """
    def __init__(self, basis, k_tab, param_atom, RWA=False):
        """Construct the free and interaction matrices on the supplied finite basis.

        Parameters
        ----------
        basis : FockBasis
            Retained occupations; q=2*i+s combines field index i with TLS label s.
        k_tab : array_like of float
            Shape (basis.modes,), signed physical momenta in occupation order.
        param_atom : dict[str, object]
            ``L``: finite positive box length; ``omega_0``: nonnegative TLS
            excitation energy; ``x_tls``: finite TLS coordinate in the phase;
            ``coupling``: 'sqrt'/'flat' form factor; ``D``: real physical
            interaction scale used when building H=H0+D*V.
        RWA : bool, optional
            True keeps a_m*sigma_+ and a_m^dagger*sigma_- only.

        Returns
        -------
        None
            Sets CSR matrices H0 and V. This eagerly allocates sparse operators;
            resource estimation should precede construction for large bases.
        """
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
        """Build the diagonal projected free operator with atomic ground energy zero.

        Parameters
        ----------
        None
            Uses this instance's basis, k_tab and param_atom['omega_0'].

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape (d,d); diagonal entry at q=2*i+s is
            E_(i,s) = sum_m |k_m|*n_(i,m) + s*omega_0, with s=0:g and s=1:e.
            No zero-point field energy or atomic -omega_0/2 offset is added.
        """
        energies = np.zeros(self.basis.dim)
        for i, occupation in enumerate(self.basis.states):
            field_energy = np.dot(np.abs(self.k_tab), occupation)
            energies[2 * i] = field_energy
            energies[2 * i + 1] = field_energy + self.param_atom["omega_0"]
        return diags(energies, format="csr")

    def interaction(self):
        """Assemble the projected interaction V, including the imposed complex phase.

        Parameters
        ----------
        None
            Uses basis, signed k_tab, L, x_tls, coupling profile and RWA.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape (d,d), excluding D. Without RWA,
            V=P[sum_m (u_m*a_m + u_m.conj()*a_m^dagger)*sigma_x]P,
            where u_m=i*f_m*exp(-i*k_m*x_tls)/sqrt(L), and f_m=sqrt(|k_m|)
            for 'sqrt', or 1 for 'flat'. P projects onto the chosen joint basis.

        Notes
        -----
        Each source column q=2*i+s and target row 2*j+(1-s) differ by one photon.
        Annihilation contributes u_m*sqrt(n_m); creation contributes
        u_m.conj()*sqrt(n_m+1). Out-of-basis targets are omitted. Under RWA,
        annihilation starts from g and creation from e, preserving sum(n_m)+s.
        COO row/column/value buffers are temporary; the returned operator is CSR.
        """
        profile = self.param_atom["coupling"]
        if profile == "sqrt":
            form_factor = np.sqrt(np.abs(self.k_tab))
        elif profile == "flat":
            form_factor = np.ones(self.basis.modes)
        else:
            raise ValueError("coupling must be flat or sqrt")
        u = 1j * form_factor * np.exp(-1j * self.k_tab * self.param_atom["x_tls"])
        u /= np.sqrt(self.param_atom["L"])
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
                        values.append(u[m] * np.sqrt(occupation[m]))
                    if not self.RWA or atom == 1:
                        target = list(occupation)
                        target[m] += 1
                        j = self.basis.index.get(tuple(target))
                        if j is not None:
                            rows.append(2 * j + 1 - atom)
                            cols.append(col)
                            values.append(u[m].conjugate() * np.sqrt(occupation[m] + 1))
        return coo_matrix((values, (rows, cols)), shape=(self.basis.dim, self.basis.dim)).tocsr()

    def build_hamiltonian(self, D=None):
        """Combine the projected operators into H=H0+D*V for evolution.

        Parameters
        ----------
        D : float or None, optional
            Finite real coupling scale. None uses param_atom['D']. This symbol is
            the physical D convention; it is not the student's historical g.

        Returns
        -------
        qutip.Qobj
            Sparse operator of shape (d,d) with flat QuTiP dimensions. Does not
            modify H0, V or the stored parameter dictionary. Real D ensures
            Hermiticity of the paired projected transitions.
        """
        D = self.param_atom["D"] if D is None else D
        if np.iscomplexobj(D) or not np.isfinite(D):
            raise ValueError("D must be finite and real")
        return qutip.Qobj(self.H0 + float(D) * self.V)
